import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import re
import torch
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

def pil_gray_loader(path):
    return Image.open(path).convert('L')

def _as_hw(image_size):
    if isinstance(image_size, (list, tuple)):
        return int(image_size[0]), int(image_size[1])
    return int(image_size), int(image_size)

def build_condition_from_heatmap(cond_path, image_size=256):
    """
    Convert raw heatmap/range condition image into a 3-channel semantic condition.

    Returns:
        cond: FloatTensor [3, H, W], range [-1, 1]
        range_mask: FloatTensor [1, H, W], range [0, 1]
    """
    h, w = _as_hw(image_size)
    img = Image.open(cond_path).convert('RGB')
    img = img.resize((w, h), BILINEAR_RESAMPLE)
    rgb = np.asarray(img, dtype=np.float32) / 255.0

    range_mask = (rgb.sum(axis=2) > 0.05).astype(np.float32)
    range_mask = binary_fill_holes(range_mask > 0).astype(np.float32)

    range_mask_uint8 = (range_mask * 255).astype(np.uint8)
    distance_map = cv2.distanceTransform(range_mask_uint8, cv2.DIST_L2, 5)
    if distance_map.max() > 0:
        distance_map = distance_map / (distance_map.max() + 1e-6)
    distance_map = distance_map.astype(np.float32)

    gray = rgb.mean(axis=2).astype(np.float32)
    kernel = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(
        range_mask.astype(np.uint8),
        kernel,
        iterations=1
    ).astype(np.float32)
    cleaned_gray = gray * inner_mask
    if cleaned_gray.max() > cleaned_gray.min():
        cleaned_gray = (
            cleaned_gray - cleaned_gray.min()
        ) / (cleaned_gray.max() - cleaned_gray.min() + 1e-6)
    cleaned_gray = cleaned_gray.astype(np.float32)

    cond = np.stack([range_mask, distance_map, cleaned_gray], axis=0).astype(np.float32)
    cond = cond * 2.0 - 1.0
    range_mask = range_mask[None, :, :].astype(np.float32)
    return torch.from_numpy(cond).float(), torch.from_numpy(range_mask).float()

def load_binary_gt(gt_path, image_size=256):
    """
    Load building layout target mask as strict binary single-channel tensor.

    Returns:
        gt: FloatTensor [1, H, W], range [-1, 1]
    """
    h, w = _as_hw(image_size)
    gt = Image.open(gt_path).convert('L')
    gt = gt.resize((w, h), NEAREST_RESAMPLE)
    gt = np.asarray(gt, dtype=np.float32)
    gt = (gt > 127).astype(np.float32)
    gt = gt * 2.0 - 1.0
    return torch.from_numpy(gt).unsqueeze(0).float()

def _save_gray_tensor(tensor, path):
    array = tensor.detach().float().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).astype(np.uint8))
    image.save(path)

def save_layout_condition_debug(sample, output_dir):
    """
    Save LayoutCondPairDataset intermediate tensors for a quick visual check.
    """
    os.makedirs(output_dir, exist_ok=True)
    cond = sample['cond_image']
    gt = sample['gt_image']

    cond_vis = (cond + 1.0) / 2.0
    gt_vis = (gt + 1.0) / 2.0

    _save_gray_tensor(cond_vis[0], os.path.join(output_dir, 'range_mask.png'))
    _save_gray_tensor(cond_vis[1], os.path.join(output_dir, 'distance_map.png'))
    _save_gray_tensor(cond_vis[2], os.path.join(output_dir, 'cleaned_gray_heatmap.png'))
    _save_gray_tensor(gt_vis[0], os.path.join(output_dir, 'gt_binary.png'))

try:
    NEAREST_RESAMPLE = Image.Resampling.NEAREST
    BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    NEAREST_RESAMPLE = Image.NEAREST
    BILINEAR_RESAMPLE = Image.BILINEAR

class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader,
                 gray_subdir='gray', image_ext='.png'):
        self.data_root = data_root
        self.gray_subdir = gray_subdir
        self.image_ext = image_ext if str(image_ext).startswith('.') else '.' + str(image_ext)
        flist = make_dataset(data_flist)
        flist = [str(x).strip() for x in flist if str(x).strip()]
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + self.image_ext

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, self.gray_subdir, file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class LayoutCondPairDataset(data.Dataset):
    """
    Pairs layout images with boundary-condition images named ``{base}_cond{ext}``.

    - ``{base}.png`` (or .jpg, etc.): ground-truth **layout** -> ``gt_image``
    - ``{base}_cond.png``: **scope / boundary** condition -> ``cond_image``

    Only pairs where both files exist are used. Files ending with ``*_cond.*``
    that have no matching layout are skipped; standalone layout files without
    a ``_cond`` partner are ignored (to avoid duplicates).
    """

    def __init__(
        self,
        data_root,
        cond_suffix='_cond',
        data_len=-1,
        image_size=[256, 256],
        layout_loader=pil_gray_loader,
        cond_loader=pil_loader,
        layout_threshold=0.5,
        range_threshold=0.03
    ):
        assert os.path.isdir(data_root), '%s is not a valid directory' % data_root
        self.data_root = data_root
        self.cond_suffix = cond_suffix if str(cond_suffix).startswith('_') else '_' + str(cond_suffix)
        self.layout_loader = layout_loader
        self.cond_loader = cond_loader
        self.image_size = image_size
        self.layout_threshold = layout_threshold
        self.range_threshold = range_threshold
        self.pil_size = (image_size[1], image_size[0])
        self.pairs = self._collect_pairs(data_root, self.cond_suffix)
        if len(self.pairs) == 0:
            raise RuntimeError(
                'LayoutCondPairDataset: no valid pairs in "{}". '
                'Expected "name.ext" (layout) and "name{}ext" (boundary).'.format(data_root, self.cond_suffix)
            )
        if data_len > 0:
            self.pairs = self.pairs[:int(data_len)]

    @staticmethod
    def _collect_pairs(root, cond_suffix):
        pat = re.compile(r'^(.+)' + re.escape(cond_suffix) + r'(\.[^.]+)$', re.IGNORECASE)
        pairs = []
        for dirpath, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                if not is_image_file(fname):
                    continue
                m = pat.match(fname)
                if not m:
                    continue
                base, ext = m.group(1), m.group(2)
                cond_path = os.path.join(dirpath, fname)
                layout_name = base + ext
                layout_path = os.path.join(dirpath, layout_name)
                if os.path.isfile(layout_path):
                    pairs.append((layout_path, cond_path))
        pairs.sort(key=lambda x: x[0])
        return pairs

    def __getitem__(self, index):
        layout_path, cond_path = self.pairs[index]
        ret = {}
        cond, range_mask = build_condition_from_heatmap(cond_path, image_size=self.image_size)
        ret['gt_image'] = load_binary_gt(layout_path, image_size=self.image_size)
        ret['cond_image'] = cond
        ret['range_mask'] = range_mask
        ret['path'] = os.path.basename(layout_path)
        return ret

    def __len__(self):
        return len(self.pairs)
