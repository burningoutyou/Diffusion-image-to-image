import csv
import os
import shutil
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from core.base_model import BaseModel
from data.dataset import build_condition_from_heatmap, is_image_file, load_binary_gt
from core.logger import LogTracker
import copy
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 

        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
            netG_for_log = self.netG.module
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
            netG_for_log = self.netG

        if getattr(netG_for_log, 'aux_loss_enabled', False):
            self.logger.info(
                'Current setting: Module 1 + Module 2B-2, lambda_bce={:.2f}, lambda_dice={:.2f}, lambda_bcr={:.2f}, use_bg_loss={}, lambda_bg={:.2f}, target_iter={}'.format(
                    netG_for_log.lambda_bce,
                    netG_for_log.lambda_dice,
                    netG_for_log.lambda_bcr,
                    getattr(netG_for_log, 'use_bg_loss', False),
                    getattr(netG_for_log, 'lambda_bg', 0.0),
                    self.opt['train'].get('n_iter', 'unknown')
                )
            )

        ''' can rewrite in inherited class for more informations logging '''
        train_metric_keys = [m.__name__ for m in losses] + [
            'loss_noise',
            'loss_bce',
            'loss_dice',
            'loss_bcr',
            'loss_bg',
            'loss_total',
            'pred_bcr_mean',
            'gt_bcr_mean',
            'learning_rate',
        ]
        self.train_metrics = LogTracker(*train_metric_keys, phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.best_loss_total = None
        self.best_epoch_sample_score = None
        self._last_train_log = {}
        self._init_epoch_sample_config()
        self._log_train_config(netG_for_log, optimizers[0])
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.range_mask = self.set_device(data.get('range_mask'))
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    @staticmethod
    def _to_01(tensor):
        tensor = tensor.detach().float().cpu()
        if tensor.numel() > 0 and float(tensor.min()) < 0.0:
            tensor = (tensor + 1.0) / 2.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _save_gray(array, path):
        array = np.clip(array, 0.0, 1.0)
        Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


    @staticmethod
    def _tensor_to_rgb_uint8(tensor):
        tensor = tensor.detach().float().cpu()
        if tensor.numel() > 0 and float(tensor.min()) < 0.0:
            tensor = (tensor + 1.0) / 2.0
        tensor = tensor.clamp(0.0, 1.0)
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if tensor.dim() != 3:
            raise ValueError('Expected CHW tensor for image saving, got {}'.format(tuple(tensor.shape)))
        array = tensor[:3].numpy()
        array = np.transpose(array, (1, 2, 0))
        return (array * 255.0).astype(np.uint8)

    @staticmethod
    def _gray_to_rgb_uint8(array):
        array = np.clip(array, 0.0, 1.0)
        gray = (array * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    @staticmethod
    def _binarize_gray_uint8(gray, binary_threshold=None):
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        if binary_threshold is None:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, int(binary_threshold), 255, cv2.THRESH_BINARY)
        return binary

    @staticmethod
    def _threshold_label(binary_threshold):
        return 'otsu' if binary_threshold is None else 't{}'.format(int(binary_threshold))

    @staticmethod
    def _parse_kernel_size(value, default=(2, 2)):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return max(1, int(value[0])), max(1, int(value[1]))
        if value is None:
            return default
        value = max(1, int(value))
        return value, value

    @staticmethod
    def _remove_small_components(binary, min_area=8):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        clean = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                clean[labels == label] = 255
        return clean

    @staticmethod
    def _clean_binary_uint8(binary, min_area=20, kernel_size=2, morphology='open', close_kernel=None):
        binary = np.clip(binary, 0, 255).astype(np.uint8)
        morphology = str(morphology or 'open').lower()
        kernel_size = max(1, int(kernel_size))

        if morphology in ['close_open_h', 'close_h_open']:
            close_w, close_h = Palette._parse_kernel_size(close_kernel, default=(5, 2))
            kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h))
            clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_h)
            kernel_close_small = np.ones((2, 2), np.uint8)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close_small)
            kernel_open = np.ones((2, 2), np.uint8)
            clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel_open)
            return Palette._remove_small_components(clean, min_area=min_area)

        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            if morphology == 'close_open':
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            else:
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return Palette._remove_small_components(binary, min_area=min_area)

    @staticmethod
    def _hole_stats(binary):
        binary = (np.clip(binary, 0, 255).astype(np.uint8) > 0).astype(np.uint8)
        if int(binary.sum()) == 0:
            return {'num_holes': 0, 'hole_pixels': 0}
        inv = (1 - binary).astype(np.uint8)
        flood = inv.copy()
        h, w = flood.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 2)
        holes = (flood == 1).astype(np.uint8)
        hole_pixels = int(holes.sum())
        if hole_pixels == 0:
            return {'num_holes': 0, 'hole_pixels': 0}
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
        num_holes = 0
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) > 0:
                num_holes += 1
        return {'num_holes': int(num_holes), 'hole_pixels': int(hole_pixels)}

    def _init_epoch_sample_config(self):
        cfg = self.opt['train'].get('epoch_sample', {}) or {}
        self.epoch_sample_cfg = cfg
        self.epoch_sample_enabled = bool(cfg.get('enabled', False))
        self.epoch_sample_pair = None
        self.epoch_sample_pairs = []
        if not self.epoch_sample_enabled:
            return

        self.epoch_sample_interval = int(cfg.get('interval', 10))
        self.epoch_sample_count = max(1, int(cfg.get('sample_count', cfg.get('num_samples', 1))))
        self.epoch_sample_save_dir = cfg.get(
            'save_dir',
            os.path.join(self.opt['path']['experiments_root'], 'epoch_sample')
        )
        self.epoch_sample_binary_threshold = cfg.get('binary_threshold', None)
        thresholds = cfg.get('binary_thresholds', [None, 100, 110, 120])
        self.epoch_sample_binary_thresholds = []
        for threshold in thresholds:
            if isinstance(threshold, str) and threshold.lower() in ['none', 'otsu', 'null']:
                threshold = None
            self.epoch_sample_binary_thresholds.append(None if threshold is None else int(threshold))
        if self.epoch_sample_binary_threshold not in self.epoch_sample_binary_thresholds:
            self.epoch_sample_binary_thresholds.insert(0, self.epoch_sample_binary_threshold)
        self.epoch_sample_clean_min_area = int(cfg.get('clean_min_area', cfg.get('min_component_area', 20)))
        self.epoch_sample_clean_kernel_size = int(cfg.get('clean_kernel_size', 2))
        self.epoch_sample_clean_morphology = cfg.get('clean_morphology', cfg.get('morphology', 'open'))
        self.epoch_sample_clean_close_kernels = cfg.get('clean_close_kernels', [[3, 2], [5, 2], [7, 2]])
        self.epoch_sample_default_clean_close_kernel = cfg.get('clean_close_kernel', [5, 2])
        self.epoch_sample_best_metric = cfg.get('best_metric', 'avg_clean_dice')
        self.epoch_sample_best_mode = cfg.get('best_mode', 'max')
        os.makedirs(self.epoch_sample_save_dir, exist_ok=True)
        self.epoch_sample_pairs = self._find_epoch_sample_pairs()
        self.epoch_sample_pair = self.epoch_sample_pairs[0] if self.epoch_sample_pairs else None
        if self.epoch_sample_pairs:
            self.logger.info('[Monitor] Fixed sample count: {}'.format(len(self.epoch_sample_pairs)))
            for idx, (target_path, cond_path) in enumerate(self.epoch_sample_pairs):
                self.logger.info('[Monitor] Fixed sample {}: {} -> {}'.format(idx, cond_path, target_path))

    def _log_train_config(self, netG_for_log, optimizer_opt):
        train_loader_args = self.opt['datasets']['train']['dataloader']['args']
        dataset_args = self.opt['datasets']['train']['which_dataset']['args']
        epoch_sample_cfg = self.opt['train'].get('epoch_sample', {}) or {}
        checkpoint_dir = self.opt['path']['checkpoint']
        sample_dir = epoch_sample_cfg.get('save_dir', os.path.join(self.opt['path']['experiments_root'], 'epoch_sample'))
        resume_state = self.opt['path'].get('resume_state')
        self.logger.info('[Train] Start training {} from epoch {}'.format(self.opt.get('name', 'model'), self.epoch))
        self.logger.info('[Train] Resume: {}'.format(bool(resume_state)))
        self.logger.info('[Config] experiment name = {}'.format(self.opt.get('name', 'unknown')))
        self.logger.info('[Config] config path = {}'.format(self.opt.get('config_path', 'unknown')))
        self.logger.info('[Config] dataset path = {}'.format(dataset_args.get('data_root')))
        self.logger.info('[Config] checkpoint save path = {}'.format(checkpoint_dir))
        self.logger.info('[Config] sample save path = {}'.format(sample_dir))
        self.logger.info('[Config] batch_size = {}'.format(train_loader_args.get('batch_size')))
        self.logger.info('[Config] num_workers = {}'.format(train_loader_args.get('num_workers')))
        self.logger.info('[Config] learning_rate = {}'.format(optimizer_opt.get('lr')))
        self.logger.info('[Config] binary_threshold = {}'.format(epoch_sample_cfg.get('binary_threshold', None)))
        self.logger.info('[Config] binary_thresholds = {}'.format(epoch_sample_cfg.get('binary_thresholds', [None, 100, 110, 120])))
        self.logger.info('[Config] min_component_area = {}'.format(epoch_sample_cfg.get('clean_min_area', epoch_sample_cfg.get('min_component_area', 20))))
        self.logger.info('[Config] clean_morphology = {}'.format(epoch_sample_cfg.get('clean_morphology', epoch_sample_cfg.get('morphology', 'open'))))
        self.logger.info('[Config] clean_close_kernels = {}'.format(epoch_sample_cfg.get('clean_close_kernels', [[3, 2], [5, 2], [7, 2]])))
        self.logger.info('[Config] target preprocessing mode = nearest_resize_strict_binary_gt')
        if getattr(netG_for_log, 'aux_loss_enabled', False):
            self.logger.info('[Config] lambda_bce = {}'.format(netG_for_log.lambda_bce))
            self.logger.info('[Config] lambda_dice = {}'.format(netG_for_log.lambda_dice))
            self.logger.info('[Config] lambda_bcr = {}'.format(netG_for_log.lambda_bcr))
            self.logger.info('[Config] use_bg_loss = {}'.format(getattr(netG_for_log, 'use_bg_loss', False)))
            self.logger.info('[Config] lambda_bg = {}'.format(getattr(netG_for_log, 'lambda_bg', 0.0)))

    def _target_from_cond_path(self, cond_path, cond_suffix):
        stem, ext = os.path.splitext(cond_path)
        if not stem.lower().endswith(cond_suffix.lower()):
            return None
        return stem[:-len(cond_suffix)] + ext

    def _find_epoch_sample_pairs(self):
        cfg = self.epoch_sample_cfg
        dataset_args = self.opt['datasets']['train']['which_dataset']['args']
        data_root = cfg.get('data_root') or dataset_args.get('data_root')
        cond_suffix = cfg.get('cond_suffix') or dataset_args.get('cond_suffix', '_cond')
        cond_suffix = cond_suffix if str(cond_suffix).startswith('_') else '_' + str(cond_suffix)
        monitor_name = cfg.get('monitor_sample_name')

        cond_files = []
        for dirpath, _, fnames in sorted(os.walk(data_root)):
            for fname in sorted(fnames):
                if not is_image_file(fname):
                    continue
                stem, _ = os.path.splitext(fname)
                if stem.lower().endswith(cond_suffix.lower()):
                    cond_files.append(os.path.join(dirpath, fname))
        if monitor_name:
            monitor_names = monitor_name if isinstance(monitor_name, list) else [monitor_name]
            monitor_set = {os.path.basename(str(name)) for name in monitor_names}
            cond_files = [p for p in cond_files if os.path.basename(p) in monitor_set]
        if not cond_files:
            raise RuntimeError('[Monitor] No condition sample found in {}'.format(data_root))

        pairs = []
        for cond_path in cond_files:
            target_path = self._target_from_cond_path(cond_path, cond_suffix)
            if target_path and os.path.isfile(target_path):
                pairs.append((target_path, cond_path))
                if len(pairs) >= self.epoch_sample_count:
                    break
        if not pairs:
            raise FileNotFoundError('[Monitor] No valid target found for condition samples in {}'.format(data_root))
        return pairs

    def _find_epoch_sample_pair(self):
        return self._find_epoch_sample_pairs()[0]

    def _load_epoch_sample_tensors(self, pair=None):
        dataset_args = self.opt['datasets']['train']['which_dataset']['args']
        image_size = dataset_args.get('image_size', [256, 256])
        target_path, cond_path = pair or self.epoch_sample_pair
        cond, range_mask = build_condition_from_heatmap(cond_path, image_size=image_size)
        target = load_binary_gt(target_path, image_size=image_size)
        return cond, target, range_mask

    @staticmethod
    def _binary_metric_row(prefix, pred_gray, target01, range01):
        pred_bin = (pred_gray > 0).astype(np.uint8)
        target_bin = (target01 > 0.5).astype(np.uint8)
        range_bin = (range01 > 0.5).astype(np.uint8)
        pred_pixels = int(pred_bin.sum())
        target_pixels = int(target_bin.sum())
        range_pixels = max(float(range_bin.sum()), 1.0)
        dice, iou = Palette._dice_iou(pred_bin, target_bin)
        precision, recall = Palette._precision_recall(pred_bin, target_bin)
        component_stats = Palette._component_geometry_stats(pred_bin)
        hole_stats = Palette._hole_stats(pred_bin * 255)
        outside_white = float((pred_bin * (1 - range_bin)).sum())
        pred_bcr = float((pred_bin * range_bin).sum() / range_pixels)
        target_bcr = float((target_bin * range_bin).sum() / range_pixels)
        row = {
            '{}_dice'.format(prefix): dice,
            '{}_iou'.format(prefix): iou,
            '{}_precision'.format(prefix): precision,
            '{}_recall'.format(prefix): recall,
            '{}_bcr_error'.format(prefix): abs(pred_bcr - target_bcr),
            '{}_pred_bcr'.format(prefix): pred_bcr,
            '{}_target_bcr'.format(prefix): target_bcr,
            '{}_white_pixels'.format(prefix): pred_pixels,
            '{}_outside_violation'.format(prefix): 0.0 if pred_pixels == 0 else outside_white / float(pred_pixels),
        }
        for key, value in component_stats.items():
            row['{}_{}'.format(prefix, key)] = value
        for key, value in hole_stats.items():
            row['{}_{}'.format(prefix, key)] = value
        return row

    @staticmethod
    def _target_connected_components(target01):
        target_bin = (target01 > 0.5).astype(np.uint8)
        return Palette._component_geometry_stats(target_bin)['connected_components']

    def _save_epoch_sample_pair(self, pair, sample_dir):
        cond, target, range_mask = self._load_epoch_sample_tensors(pair)
        cond_batch = self.set_device(cond.unsqueeze(0))
        netG = self.netG.module if self.opt['distributed'] else self.netG
        with torch.no_grad():
            output, _ = netG.restoration(cond_batch, sample_num=self.sample_num)

        pred01 = self._to_01(output)[0, 0].numpy()
        range01 = range_mask[0].detach().float().cpu().numpy()
        pred01 = pred01 * (range01 > 0.5).astype(np.float32)
        target01 = self._to_01(target)[0].numpy()
        input_rgb = self._tensor_to_rgb_uint8(cond)
        generated_raw_rgb = self._gray_to_rgb_uint8(pred01)
        generated_raw_gray = np.clip(pred01 * 255.0, 0, 255).astype(np.uint8)
        range_binary_gray = ((range01 > 0.5).astype(np.uint8) * 255)
        threshold_binaries = {}
        for threshold in self.epoch_sample_binary_thresholds:
            label = self._threshold_label(threshold)
            binary_gray = self._binarize_gray_uint8(
                generated_raw_gray,
                binary_threshold=threshold
            )
            threshold_binaries[label] = np.where(range_binary_gray > 0, binary_gray, 0).astype(np.uint8)
        main_binary_label = self._threshold_label(self.epoch_sample_binary_threshold)
        generated_binary_gray = threshold_binaries[main_binary_label]
        generated_binary_rgb = np.repeat(generated_binary_gray[:, :, None], 3, axis=2)

        clean_variants = {}
        for kernel_pair in self.epoch_sample_clean_close_kernels:
            close_w, close_h = self._parse_kernel_size(kernel_pair, default=(5, 2))
            label = 'close_{}x{}'.format(close_w, close_h)
            clean_gray = self._clean_binary_uint8(
                generated_binary_gray,
                min_area=self.epoch_sample_clean_min_area,
                kernel_size=self.epoch_sample_clean_kernel_size,
                morphology='close_open_h',
                close_kernel=(close_w, close_h)
            )
            clean_variants[label] = np.where(range_binary_gray > 0, clean_gray, 0).astype(np.uint8)
        default_w, default_h = self._parse_kernel_size(self.epoch_sample_default_clean_close_kernel, default=(5, 2))
        default_clean_label = 'close_{}x{}'.format(default_w, default_h)
        if default_clean_label not in clean_variants:
            clean_gray = self._clean_binary_uint8(
                generated_binary_gray,
                min_area=self.epoch_sample_clean_min_area,
                kernel_size=self.epoch_sample_clean_kernel_size,
                morphology='close_open_h',
                close_kernel=(default_w, default_h)
            )
            clean_variants[default_clean_label] = np.where(range_binary_gray > 0, clean_gray, 0).astype(np.uint8)
        generated_clean_gray = clean_variants[default_clean_label]
        generated_clean_rgb = np.repeat(generated_clean_gray[:, :, None], 3, axis=2)
        target_rgb = self._gray_to_rgb_uint8(target01)

        os.makedirs(sample_dir, exist_ok=True)
        Image.fromarray(input_rgb).save(os.path.join(sample_dir, 'input.png'))
        Image.fromarray(target_rgb).save(os.path.join(sample_dir, 'target.png'))
        Image.fromarray(generated_raw_rgb).save(os.path.join(sample_dir, 'generated_raw.png'))
        Image.fromarray(generated_binary_rgb).save(os.path.join(sample_dir, 'generated_binary.png'))
        for label, binary_gray in threshold_binaries.items():
            binary_rgb = np.repeat(binary_gray[:, :, None], 3, axis=2)
            Image.fromarray(binary_rgb).save(os.path.join(sample_dir, 'generated_binary_{}.png'.format(label)))
        Image.fromarray(generated_clean_rgb).save(os.path.join(sample_dir, 'generated_clean.png'))
        for label, clean_gray in clean_variants.items():
            clean_rgb = np.repeat(clean_gray[:, :, None], 3, axis=2)
            Image.fromarray(clean_rgb).save(os.path.join(sample_dir, 'generated_clean_{}.png'.format(label)))
            Image.fromarray(np.concatenate([input_rgb, clean_rgb, target_rgb], axis=1)).save(
                os.path.join(sample_dir, 'compare_clean_{}.png'.format(label)))
        Image.fromarray(np.concatenate([input_rgb, generated_raw_rgb, target_rgb], axis=1)).save(
            os.path.join(sample_dir, 'compare_raw.png'))
        Image.fromarray(np.concatenate([input_rgb, generated_binary_rgb, target_rgb], axis=1)).save(
            os.path.join(sample_dir, 'compare_binary.png'))
        Image.fromarray(np.concatenate([input_rgb, generated_clean_rgb, target_rgb], axis=1)).save(
            os.path.join(sample_dir, 'compare_clean.png'))
        Image.fromarray(np.concatenate([input_rgb, generated_raw_rgb, generated_binary_rgb, generated_clean_rgb, target_rgb], axis=1)).save(
            os.path.join(sample_dir, 'compare_all.png'))
        Image.fromarray(generated_raw_rgb).save(os.path.join(sample_dir, 'generated.png'))
        Image.fromarray(np.concatenate([input_rgb, generated_raw_rgb, target_rgb], axis=1)).save(
            os.path.join(sample_dir, 'compare.png'))

        target_path, cond_path = pair
        target_connected = self._target_connected_components(target01)
        row = {
            'sample_id': os.path.splitext(os.path.basename(target_path))[0],
            'name': os.path.basename(target_path),
            'cond_path': cond_path,
            'target_path': target_path,
            'output_dir': sample_dir,
            'binary_threshold_label': main_binary_label,
            'clean_variant': default_clean_label,
            'raw_mean': float(generated_raw_gray.mean()),
            'raw_std': float(generated_raw_gray.std()),
            'raw_min': int(generated_raw_gray.min()),
            'raw_max': int(generated_raw_gray.max()),
            'target_white_pixels': int((target01 > 0.5).sum()),
            'target_connected_components': target_connected,
        }
        row.update(self._binary_metric_row('binary', generated_binary_gray, target01, range01))
        row.update(self._binary_metric_row('clean', generated_clean_gray, target01, range01))
        row['component_count_error'] = abs(row['clean_connected_components'] - target_connected)
        row['max_component_area_ratio'] = row['clean_max_component_area_ratio']
        row['outside_violation'] = row['clean_outside_violation']
        row['hole_pixels'] = row['clean_hole_pixels']
        row['num_holes'] = row['clean_num_holes']
        return row

    def _write_epoch_sample_metrics(self, rows, metrics_dir):
        if not rows:
            return {}
        os.makedirs(metrics_dir, exist_ok=True)
        fieldnames = list(rows[0].keys())
        for filename in ['multi_sample_summary.csv', 'multi_sample_metrics.csv']:
            with open(os.path.join(metrics_dir, filename), 'w', newline='', encoding='utf-8-sig') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        numeric_keys = []
        for key, value in rows[0].items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_keys.append(key)
        overview = {'num_samples': len(rows)}
        for key in numeric_keys:
            overview['avg_{}'.format(key)] = float(np.mean([float(row[key]) for row in rows]))

        overview_path = os.path.join(metrics_dir, 'multi_sample_metrics_overview.csv')
        with open(overview_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(overview.keys()))
            writer.writeheader()
            writer.writerow(overview)
        return overview

    def _maybe_update_epoch_sample_best(self, overview):
        if self.opt['global_rank'] != 0 or not overview:
            return
        metric = self.epoch_sample_best_metric
        if metric not in overview:
            self.logger.warning('[Best] Metric {} not found in epoch sample overview.'.format(metric))
            return
        score = float(overview[metric])
        is_better = self.best_epoch_sample_score is None
        if not is_better:
            if self.epoch_sample_best_mode == 'min':
                is_better = score < self.best_epoch_sample_score
            else:
                is_better = score > self.best_epoch_sample_score
        if not is_better:
            return

        self.best_epoch_sample_score = score
        checkpoint_dir = self.opt['path']['checkpoint']
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(self._network_state_dict_cpu(self.netG), best_model_path)
        if self.ema_scheduler is not None:
            torch.save(self._network_state_dict_cpu(self.netG_EMA), os.path.join(checkpoint_dir, 'best_model_ema.pth'))
        best_txt = os.path.join(checkpoint_dir, 'best_checkpoint.txt')
        with open(best_txt, 'w', encoding='utf-8') as f:
            f.write('best_epoch={}\n'.format(self.epoch))
            f.write('best_{}={}\n'.format(metric, score))
            f.write('best_checkpoint_path={}\n'.format(best_model_path))
        self.logger.info('[Best] New best model at epoch {}, {} = {:.6f}'.format(self.epoch, metric, score))

    def _run_epoch_sample(self):
        if not self.epoch_sample_enabled or not self.epoch_sample_pairs:
            return
        if self.epoch % self.epoch_sample_interval != 0:
            return

        was_training = self.netG.training
        self.netG.eval()
        epoch_dir = os.path.join(self.epoch_sample_save_dir, 'epoch_{:04d}'.format(self.epoch))
        os.makedirs(epoch_dir, exist_ok=True)
        metrics_dir = epoch_dir
        if len(self.epoch_sample_pairs) > 1:
            metrics_dir = os.path.join(epoch_dir, 'multi_samples')
            os.makedirs(metrics_dir, exist_ok=True)
        rows = []
        for idx, pair in enumerate(self.epoch_sample_pairs):
            if len(self.epoch_sample_pairs) == 1:
                sample_dir = epoch_dir
            else:
                target_path, _ = pair
                sample_name = os.path.splitext(os.path.basename(target_path))[0]
                sample_dir = os.path.join(metrics_dir, 'sample_{:03d}_{}'.format(idx, sample_name))
            rows.append(self._save_epoch_sample_pair(pair, sample_dir))
        self.netG.train(was_training)

        overview = self._write_epoch_sample_metrics(rows, metrics_dir)
        self._maybe_update_epoch_sample_best(overview)
        self.logger.info('[Monitor] Epoch {} saved {} sample(s) to: {}'.format(
            self.epoch, len(rows), metrics_dir))

    def on_train_epoch_end(self):
        self._run_epoch_sample()

    @staticmethod
    def _component_geometry_stats(binary, min_area=10, small_area=20):
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=8
        )
        total_area = float(binary.astype(np.uint8).sum())
        if total_area <= 0:
            return {
                'connected_components': 0,
                'avg_component_area': 0.0,
                'small_fragment_ratio': 1.0,
                'max_component_area_ratio': 1.0,
                'aspect_ratio_mean': 0.0,
                'aspect_ratio_valid_ratio': 0.0,
            }

        valid_areas = []
        valid_aspect_ratios = []
        small_fragment_area = 0.0
        max_area = 0.0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            max_area = max(max_area, float(area))
            if area < small_area:
                small_fragment_area += float(area)
            if area < min_area:
                continue
            width = float(stats[label, cv2.CC_STAT_WIDTH])
            height = float(stats[label, cv2.CC_STAT_HEIGHT])
            aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
            valid_areas.append(area)
            valid_aspect_ratios.append(aspect_ratio)

        if not valid_areas:
            return {
                'connected_components': 0,
                'avg_component_area': 0.0,
                'small_fragment_ratio': float(small_fragment_area / total_area),
                'max_component_area_ratio': float(max_area / total_area),
                'aspect_ratio_mean': 0.0,
                'aspect_ratio_valid_ratio': 0.0,
            }

        aspect_ratios = np.asarray(valid_aspect_ratios, dtype=np.float32)
        return {
            'connected_components': len(valid_areas),
            'avg_component_area': float(np.mean(valid_areas)),
            'small_fragment_ratio': float(small_fragment_area / total_area),
            'max_component_area_ratio': float(max_area / total_area),
            'aspect_ratio_mean': float(aspect_ratios.mean()),
            'aspect_ratio_valid_ratio': float((aspect_ratios > 2.0).mean()),
        }

    @staticmethod
    def _dice_iou(pred, gt):
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        union = np.logical_or(pred, gt).sum()
        dice_den = pred_sum + gt_sum
        dice = 1.0 if dice_den == 0 else float(2.0 * intersection / dice_den)
        iou = 1.0 if union == 0 else float(intersection / union)
        return dice, iou

    @staticmethod
    def _precision_recall(pred, gt):
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        true_positive = np.logical_and(pred, gt).sum()
        false_positive = np.logical_and(pred, np.logical_not(gt)).sum()
        false_negative = np.logical_and(np.logical_not(pred), gt).sum()
        precision = float(true_positive / (true_positive + false_positive + 1e-6))
        recall = float(true_positive / (true_positive + false_negative + 1e-6))
        return precision, recall

    def _get_range_mask_batch(self):
        if self.range_mask is not None:
            return self._to_01(self.range_mask)
        return ((self.cond_image.detach().float().cpu()[:, 0:1] + 1.0) / 2.0).clamp(0.0, 1.0)

    def _save_layout_eval_outputs(self, phase):
        if self.opt['global_rank'] != 0:
            return []

        result_root = os.path.join(self.opt['path']['results'], phase, str(self.epoch))
        eval_root = os.path.join(result_root, 'layout_eval')
        os.makedirs(eval_root, exist_ok=True)

        pred_batch = self._to_01(self.output)
        gt_batch = self._to_01(self.gt_image)
        range_batch = self._get_range_mask_batch()
        thresholds = [0.5, 0.6, 0.7]
        rows = []

        for idx in range(self.batch_size):
            name = os.path.basename(self.path[idx])
            stem, _ = os.path.splitext(name)
            sample_dir = os.path.join(eval_root, stem)
            os.makedirs(sample_dir, exist_ok=True)

            pred01 = pred_batch[idx, 0].numpy()
            gt01 = gt_batch[idx, 0].numpy()
            range01 = range_batch[idx, 0].numpy()
            gt_bin = (gt01 > 0.5).astype(np.uint8)
            range_bin = (range01 > 0.5).astype(np.uint8)
            range_pixels = max(float(range_bin.sum()), 1.0)

            self._save_gray(pred01, os.path.join(sample_dir, 'raw_output.png'))
            self._save_gray(gt_bin.astype(np.float32), os.path.join(sample_dir, 'gt.png'))
            self._save_gray(range_bin.astype(np.float32), os.path.join(sample_dir, 'range_mask.png'))

            row = {
                'path': name,
                'GT_BCR': float((gt_bin * range_bin).sum() / range_pixels),
            }

            for threshold in thresholds:
                key = '{:.1f}'.format(threshold)
                binary_raw = (pred01 > threshold).astype(np.uint8)
                binary_masked = (binary_raw * range_bin).astype(np.uint8)
                self._save_gray(binary_masked.astype(np.float32), os.path.join(sample_dir, 'binary_{}.png'.format(key)))
                if key == '0.5':
                    self._save_gray(binary_masked.astype(np.float32), os.path.join(sample_dir, 'selected_binary.png'))

                pred_total = float(binary_raw.sum())
                outside_white = float((binary_raw * (1 - range_bin)).sum())
                component_stats = self._component_geometry_stats(binary_masked)
                dice, iou = self._dice_iou(binary_masked, gt_bin)
                precision, recall = self._precision_recall(binary_masked, gt_bin)

                row['Pred_BCR_{}'.format(key)] = float(binary_masked.sum() / range_pixels)
                row['outside_violation_{}'.format(key)] = 0.0 if pred_total == 0 else outside_white / pred_total
                row['connected_components_{}'.format(key)] = component_stats['connected_components']
                row['avg_component_area_{}'.format(key)] = component_stats['avg_component_area']
                row['max_component_area_ratio_{}'.format(key)] = component_stats['max_component_area_ratio']
                row['aspect_ratio_valid_ratio_{}'.format(key)] = component_stats['aspect_ratio_valid_ratio']
                row['small_fragment_ratio_{}'.format(key)] = component_stats['small_fragment_ratio']
                row['dice_{}'.format(key)] = dice
                row['iou_{}'.format(key)] = iou
                row['precision_{}'.format(key)] = precision
                row['recall_{}'.format(key)] = recall

            rows.append(row)
        return rows

    def _save_layout_metrics_csv(self, phase, rows):
        if self.opt['global_rank'] != 0 or not rows:
            return
        result_root = os.path.join(self.opt['path']['results'], phase, str(self.epoch))
        os.makedirs(result_root, exist_ok=True)
        fieldnames = [
            'path',
            'GT_BCR',
            'Pred_BCR_0.5',
            'Pred_BCR_0.6',
            'Pred_BCR_0.7',
            'outside_violation_0.5',
            'outside_violation_0.6',
            'outside_violation_0.7',
            'connected_components_0.5',
            'connected_components_0.6',
            'connected_components_0.7',
            'avg_component_area_0.5',
            'avg_component_area_0.6',
            'avg_component_area_0.7',
            'max_component_area_ratio_0.5',
            'max_component_area_ratio_0.6',
            'max_component_area_ratio_0.7',
            'aspect_ratio_valid_ratio_0.5',
            'aspect_ratio_valid_ratio_0.6',
            'aspect_ratio_valid_ratio_0.7',
            'small_fragment_ratio_0.5',
            'small_fragment_ratio_0.6',
            'small_fragment_ratio_0.7',
            'dice_0.5',
            'dice_0.6',
            'dice_0.7',
            'iou_0.5',
            'iou_0.6',
            'iou_0.7',
            'precision_0.5',
            'precision_0.6',
            'precision_0.7',
            'recall_0.5',
            'recall_0.6',
            'recall_0.7',
        ]
        with open(os.path.join(result_root, 'metrics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(
                self.gt_image,
                self.cond_image,
                range_mask=self.range_mask
            )
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            netG = self.netG.module if self.opt['distributed'] else self.netG
            for key, value in netG.get_loss_details().items():
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                self.train_metrics.update(key, float(value))
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
            if self.iter >= self.opt['train']['n_iter']:
                break

        current_lr = self.optG.param_groups[0].get('lr', 0.0)
        self.train_metrics.update('learning_rate', float(current_lr))
        for scheduler in self.schedulers:
            scheduler.step()
        result = self.train_metrics.result()
        self._last_train_log = dict(result)
        return result
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        layout_rows = []
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
                layout_rows.extend(self._save_layout_eval_outputs(phase='val'))

        self._save_layout_metrics_csv('val', layout_rows)
        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        layout_rows = []
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)

                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
                layout_rows.extend(self._save_layout_eval_outputs(phase='test'))
        
        self._save_layout_metrics_csv('test', layout_rows)
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)


    def _network_state_dict_cpu(self, network):
        if isinstance(network, torch.nn.DataParallel) or isinstance(network, torch.nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        return {key: value.cpu() for key, value in state_dict.items()}

    def _training_state_dict(self):
        return {
            'epoch': self.epoch,
            'iter': self.iter,
            'schedulers': [s.state_dict() for s in self.schedulers],
            'optimizers': [o.state_dict() for o in self.optimizers],
        }

    def _save_checkpoint_aliases(self, netG_label):
        if self.opt['global_rank'] != 0:
            return
        train_opt = self.opt['train']
        save_latest = bool(train_opt.get('save_latest_alias', False))
        save_best = bool(train_opt.get('save_best_alias', False))
        if not save_latest and not save_best:
            return

        checkpoint_dir = self.opt['path']['checkpoint']
        current_loss = self._last_train_log.get('loss_total')
        if current_loss is None:
            current_loss = self._last_train_log.get(self.loss_fn.__name__)
        current_loss = float(current_loss) if current_loss is not None else None

        aliases = []
        if save_latest:
            aliases.append('latest')
        if save_best and current_loss is not None:
            if self.best_loss_total is None or current_loss < self.best_loss_total:
                self.best_loss_total = current_loss
                aliases.append('best')

        for alias in aliases:
            torch.save(
                self._network_state_dict_cpu(self.netG),
                os.path.join(checkpoint_dir, '{}_{}.pth'.format(alias, netG_label))
            )
            if self.ema_scheduler is not None:
                torch.save(
                    self._network_state_dict_cpu(self.netG_EMA),
                    os.path.join(checkpoint_dir, '{}_{}.pth'.format(alias, netG_label + '_ema'))
                )
            torch.save(
                self._training_state_dict(),
                os.path.join(checkpoint_dir, '{}.state'.format(alias))
            )

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()
        self._save_checkpoint_aliases(netG_label)


