import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_condition_from_heatmap, load_binary_gt
from data.dataset import save_layout_condition_debug


def main():
    parser = argparse.ArgumentParser(description='Visualize LayoutCondPairDataset condition re-encoding.')
    parser.add_argument('cond_path', help='Path to xxx_cond.png')
    parser.add_argument('--gt-path', required=True, help='Path to the paired GT mask')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--output-dir', default='condition_reencoding_vis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cond, range_mask = build_condition_from_heatmap(args.cond_path, args.image_size)
    sample = {
        'cond_image': cond,
        'gt_image': load_binary_gt(args.gt_path, args.image_size),
        'range_mask': range_mask,
    }
    save_layout_condition_debug(sample, output_dir)

    image = Image.fromarray((range_mask[0].numpy() * 255.0).astype(np.uint8))
    image.save(output_dir / 'range_mask_raw.png')

    print('Saved condition visualizations to {}'.format(output_dir.resolve()))


if __name__ == '__main__':
    main()
