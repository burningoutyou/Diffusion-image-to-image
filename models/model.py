import csv
import os
import cv2
import numpy as np
import torch
import tqdm
from PIL import Image
from core.base_model import BaseModel
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
                'Current setting: Module 1 + Module 2B-2, lambda_bce={:.2f}, lambda_dice={:.2f}, lambda_bcr={:.2f}, target_iter={}'.format(
                    netG_for_log.lambda_bce,
                    netG_for_log.lambda_dice,
                    netG_for_log.lambda_bcr,
                    self.opt['train'].get('n_iter', 'unknown')
                )
            )

        ''' can rewrite in inherited class for more informations logging '''
        train_metric_keys = [m.__name__ for m in losses] + [
            'loss_noise',
            'loss_bce',
            'loss_dice',
            'loss_bcr',
            'loss_total',
            'pred_bcr_mean',
            'gt_bcr_mean',
        ]
        self.train_metrics = LogTracker(*train_metric_keys, phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.range_mask = self.set_device(data.get('range_mask'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
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
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

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
                mask=self.mask,
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

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        layout_rows = []
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
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
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
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
