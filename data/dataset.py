import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import re
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

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

    def _load_layout_mask(self, path):
        layout = self.layout_loader(path).resize(self.pil_size, NEAREST_RESAMPLE)
        layout = np.asarray(layout, dtype=np.float32) / 255.0
        layout = (layout > self.layout_threshold).astype(np.float32)
        return torch.from_numpy(layout).unsqueeze(0) * 2.0 - 1.0

    def _load_condition_map(self, path):
        cond = self.cond_loader(path).resize(self.pil_size, BILINEAR_RESAMPLE)
        cond = np.asarray(cond, dtype=np.float32) / 255.0

        range_mask = (cond.max(axis=2) > self.range_threshold).astype(np.float32)

        distance_map = distance_transform_edt(range_mask)
        max_distance = distance_map.max()
        if max_distance > 0:
            distance_map = distance_map / max_distance

        gray_heatmap = 0.299 * cond[:, :, 0] + 0.587 * cond[:, :, 1] + 0.114 * cond[:, :, 2]
        gray_heatmap = gray_heatmap * range_mask
        max_gray = gray_heatmap.max()
        if max_gray > 0:
            gray_heatmap = gray_heatmap / max_gray

        cond = np.stack([range_mask, distance_map, gray_heatmap], axis=0).astype(np.float32)
        return torch.from_numpy(cond) * 2.0 - 1.0

    def __getitem__(self, index):
        layout_path, cond_path = self.pairs[index]
        ret = {}
        ret['gt_image'] = self._load_layout_mask(layout_path)
        ret['cond_image'] = self._load_condition_map(cond_path)
        ret['path'] = os.path.basename(layout_path)
        return ret

    def __len__(self):
        return len(self.pairs)
