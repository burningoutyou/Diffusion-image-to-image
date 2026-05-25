import math
import torch
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', aux_loss=None, **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.aux_loss_opt = aux_loss or {}
        self.aux_loss_enabled = bool(self.aux_loss_opt.get('enabled', False))
        # Fixed Module 2B-2: BCE=0.04, Dice=0.05, BCR=0.01.
        self.lambda_bce = float(self.aux_loss_opt.get('lambda_bce', 0.04))
        self.lambda_dice = float(self.aux_loss_opt.get('lambda_dice', 0.05))
        self.lambda_bcr = float(self.aux_loss_opt.get('lambda_bcr', 0.01))
        self.loss_details = {}

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def predict_start_from_sampled_noise(self, y_t, sample_gammas, noise):
        sample_gammas = sample_gammas.view(-1, 1, 1, 1)
        return (
            y_t - (1.0 - sample_gammas).sqrt() * noise
        ) / (sample_gammas.sqrt() + 1e-8)

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    def is_layout_condition(self, y_cond, sample):
        return (
            y_cond is not None and sample is not None and
            y_cond.dim() == 4 and sample.dim() == 4 and
            y_cond.shape[1] == 3 and sample.shape[1] == 1
        )

    def apply_final_range_mask(self, sample, y_cond):
        # LayoutCondPairDataset stores range_mask as condition channel 0 in [-1, 1].
        range_mask = (y_cond[:, 0:1] > 0.0).float()
        background = torch.full_like(sample, -1.0)
        return torch.where(range_mask > 0.5, sample, background)

    def get_loss_details(self):
        return dict(self.loss_details)

    def get_range_mask_for_aux_loss(self, y_cond, range_mask):
        if range_mask is not None:
            return range_mask.float().clamp(0.0, 1.0)
        if y_cond is not None and y_cond.dim() == 4 and y_cond.shape[1] >= 1:
            return ((y_cond[:, 0:1].float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return None

    def x0_auxiliary_loss(self, x0_pred, y_0, y_cond=None, range_mask=None):
        range_mask = self.get_range_mask_for_aux_loss(y_cond, range_mask)
        zero = x0_pred.new_tensor(0.0)
        if range_mask is None:
            return zero, {
                'loss_bce': zero,
                'loss_dice': zero,
                'loss_bcr': zero,
                'pred_bcr_mean': zero,
                'gt_bcr_mean': zero,
            }

        pred01 = ((x0_pred + 1.0) / 2.0).clamp(1e-4, 1.0 - 1e-4)
        gt01 = ((y_0 + 1.0) / 2.0).clamp(0.0, 1.0)
        range_mask = range_mask.to(device=pred01.device, dtype=pred01.dtype)

        loss_bce = F.binary_cross_entropy(pred01, gt01)

        dims = [1, 2, 3]
        intersection = (pred01 * gt01).sum(dim=dims)
        union = pred01.sum(dim=dims) + gt01.sum(dim=dims)
        loss_dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()

        range_area = range_mask.sum(dim=dims) + 1e-6
        pred_bcr = (pred01 * range_mask).sum(dim=dims) / range_area
        gt_bcr = (gt01 * range_mask).sum(dim=dims) / range_area
        loss_bcr = F.l1_loss(pred_bcr, gt_bcr)

        loss_aux = (
            self.lambda_bce * loss_bce +
            self.lambda_dice * loss_dice +
            self.lambda_bcr * loss_bcr
        )
        return loss_aux, {
            'loss_bce': loss_bce,
            'loss_dice': loss_dice,
            'loss_bcr': loss_bcr,
            'pred_bcr_mean': pred_bcr.mean(),
            'gt_bcr_mean': gt_bcr.mean(),
        }

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, _, h, w = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)

        out_channel = self.denoise_fn.out_channel
        y_t = default(y_t, lambda: torch.randn(
            b, out_channel, h, w, device=y_cond.device, dtype=y_cond.dtype
        ))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        if self.is_layout_condition(y_cond, y_t):
            y_t = self.apply_final_range_mask(y_t, y_cond)
            ret_arr = ret_arr.clone()
            ret_arr[-b:] = y_t
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None, range_mask=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss_noise = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss_noise = self.loss_fn(noise, noise_hat)

        loss_total = loss_noise
        zero = loss_noise.detach().new_tensor(0.0)
        aux_details = {
            'loss_bce': zero,
            'loss_dice': zero,
            'loss_bcr': zero,
            'pred_bcr_mean': zero,
            'gt_bcr_mean': zero,
        }
        if self.aux_loss_enabled:
            x0_pred = self.predict_start_from_sampled_noise(
                y_noisy, sample_gammas, noise_hat
            ).clamp(-1.0, 1.0)
            loss_aux, aux_details = self.x0_auxiliary_loss(
                x0_pred, y_0, y_cond=y_cond, range_mask=range_mask
            )
            loss_total = loss_total + loss_aux

        self.loss_details = {
            'loss_noise': loss_noise.detach(),
            'loss_bce': aux_details['loss_bce'].detach(),
            'loss_dice': aux_details['loss_dice'].detach(),
            'loss_bcr': aux_details['loss_bcr'].detach(),
            'loss_total': loss_total.detach(),
            'pred_bcr_mean': aux_details['pred_bcr_mean'].detach(),
            'gt_bcr_mean': aux_details['gt_bcr_mean'].detach(),
        }
        return loss_total


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
