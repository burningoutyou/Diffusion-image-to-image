import argparse
import csv
import json
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from multi_seed_candidate_validator import (  # noqa: E402
    StdLogger,
    build_dataset,
    build_network,
    load_json_with_comments,
    restoration_with_candidate_seeds,
    restoration_with_fast_seed_batch,
)


IMAGE_NAMES = [
    "raw_output.png",
    "selected_binary.png",
    "binary_0.5.png",
    "binary_0.6.png",
    "binary_0.7.png",
    "range_mask.png",
    "condition_vis.png",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def to_01(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.numel() > 0 and float(tensor.min()) < 0.0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def save_gray(array, path):
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def save_rgb(array, path):
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def read_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    ensure_dir(os.path.dirname(path))
    if fieldnames is None:
        fieldnames = fieldnames_for(rows)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fieldnames_for(rows):
    names = []
    for row in rows:
        for key in row.keys():
            if key not in names:
                names.append(key)
    return names


def to_float(row, key, default=0.0):
    value = row.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, default)
    if value is None or value == "":
        return int(default)
    return int(float(value))


def sample_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def component_metrics(binary, min_area=10, small_area=20):
    binary_u8 = binary.astype(np.uint8)
    total_area = float(binary_u8.sum())
    if total_area <= 0.0:
        return {
            "connected_components": 0,
            "avg_component_area": 0.0,
            "small_fragment_ratio": 1.0,
            "max_component_area_ratio": 1.0,
            "aspect_ratio_mean": 0.0,
            "aspect_ratio_valid_ratio": 0.0,
        }

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    valid_areas = []
    valid_ratios = []
    small_fragment_area = 0.0
    all_areas = []

    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        width = float(stats[label, cv2.CC_STAT_WIDTH])
        height = float(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= 0.0:
            continue
        all_areas.append(area)
        if area < small_area:
            small_fragment_area += area
        if area >= min_area:
            valid_areas.append(area)
            ratio = max(width, height) / (min(width, height) + 1e-6)
            valid_ratios.append(float(ratio))

    if not valid_areas:
        return {
            "connected_components": 0,
            "avg_component_area": 0.0,
            "small_fragment_ratio": float(small_fragment_area / total_area),
            "max_component_area_ratio": float(max(all_areas) / total_area) if all_areas else 1.0,
            "aspect_ratio_mean": 0.0,
            "aspect_ratio_valid_ratio": 0.0,
        }

    return {
        "connected_components": len(valid_areas),
        "avg_component_area": float(np.mean(valid_areas)),
        "small_fragment_ratio": float(small_fragment_area / total_area),
        "max_component_area_ratio": float(max(all_areas) / total_area) if all_areas else 1.0,
        "aspect_ratio_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "aspect_ratio_valid_ratio": float(
            sum(1 for ratio in valid_ratios if ratio > 2.0) / len(valid_ratios)
        ) if valid_ratios else 0.0,
    }


def score_standard(row, target_bcr=0.22, target_components=4.0):
    return (
        2.0 * abs(to_float(row, "Pred_BCR") - target_bcr)
        + 1.0 * abs(to_float(row, "connected_components") - target_components) / target_components
        + 2.0 * to_float(row, "outside_violation")
        + 1.0 * to_float(row, "small_fragment_ratio")
        + 1.0 * max(0.0, to_float(row, "max_component_area_ratio") - 0.50)
        + 1.0 * max(0.0, 0.50 - to_float(row, "aspect_ratio_valid_ratio"))
    )


def score_strict(row, target_bcr=0.22, target_components=4.0):
    return (
        2.0 * abs(to_float(row, "Pred_BCR") - target_bcr)
        + 1.0 * abs(to_float(row, "connected_components") - target_components) / target_components
        + 2.0 * to_float(row, "outside_violation")
        + 1.0 * to_float(row, "small_fragment_ratio")
        + 1.5 * max(0.0, to_float(row, "max_component_area_ratio") - 0.45)
        + 1.5 * max(0.0, 0.60 - to_float(row, "aspect_ratio_valid_ratio"))
    )


def metrics_from_prediction(pred01, range01, threshold=0.5):
    range_bin = (range01 > 0.5).astype(np.uint8)
    range_pixels = max(float(range_bin.sum()), 1.0)
    binary_raw = (pred01 > threshold).astype(np.uint8)
    binary_masked = (binary_raw * range_bin).astype(np.uint8)
    pred_total = float(binary_raw.sum())
    outside_white = float((binary_raw * (1 - range_bin)).sum())

    row = OrderedDict()
    row["Pred_BCR"] = float(binary_masked.sum() / range_pixels)
    row["outside_violation"] = 0.0 if pred_total == 0.0 else float(outside_white / pred_total)
    row.update(component_metrics(binary_masked))
    return row, binary_masked


def metrics_from_existing_row(row):
    out = OrderedDict()
    for key in [
        "Pred_BCR",
        "outside_violation",
        "connected_components",
        "avg_component_area",
        "small_fragment_ratio",
        "max_component_area_ratio",
        "aspect_ratio_mean",
        "aspect_ratio_valid_ratio",
    ]:
        out[key] = to_float(row, key)
    out["connected_components"] = int(round(out["connected_components"]))
    return out


def candidate_dir(sample_dir, candidate_id):
    return os.path.join(sample_dir, "candidate_{:02d}".format(candidate_id))


def copy_candidate(src_dir, dst_dir):
    ensure_dir(dst_dir)
    if not src_dir or not os.path.isdir(src_dir):
        return
    for name in IMAGE_NAMES:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def save_candidate_images(dst_dir, pred01, range01, cond01):
    ensure_dir(dst_dir)
    save_gray(pred01, os.path.join(dst_dir, "raw_output.png"))
    save_gray(range01, os.path.join(dst_dir, "range_mask.png"))
    save_rgb(np.transpose(cond01, (1, 2, 0)), os.path.join(dst_dir, "condition_vis.png"))
    for threshold in [0.5, 0.6, 0.7]:
        _, binary = metrics_from_prediction(pred01, range01, threshold=threshold)
        save_gray(binary.astype(np.float32), os.path.join(dst_dir, "binary_{:.1f}.png".format(threshold)))
    shutil.copy2(
        os.path.join(dst_dir, "binary_0.5.png"),
        os.path.join(dst_dir, "selected_binary.png"),
    )


def add_scores_and_ranks(rows, target_bcr, target_components):
    for row in rows:
        row["standard_validator_score"] = score_standard(row, target_bcr, target_components)
        row["strict_validator_score"] = score_strict(row, target_bcr, target_components)

    for rank, row in enumerate(
        sorted(rows, key=lambda item: (to_float(item, "standard_validator_score"), to_int(item, "candidate_id"))),
        start=1,
    ):
        row["standard_rank"] = rank
    for rank, row in enumerate(
        sorted(rows, key=lambda item: (to_float(item, "strict_validator_score"), to_int(item, "candidate_id"))),
        start=1,
    ):
        row["strict_rank"] = rank
    return sorted(rows, key=lambda item: (to_int(item, "strict_rank"), to_int(item, "candidate_id")))


def write_top_k(sample_dir, rows, top_k):
    top_dir = ensure_dir(os.path.join(sample_dir, "top_k_candidates"))
    for row in rows[:top_k]:
        candidate_id = to_int(row, "candidate_id")
        src_dir = candidate_dir(sample_dir, candidate_id)
        dst_dir = os.path.join(
            top_dir,
            "rank_{:02d}_candidate_{:02d}".format(to_int(row, "strict_rank"), candidate_id),
        )
        copy_candidate(src_dir, dst_dir)


def load_reuse_rows(reuse_sample_dir, formal_sample_dir, num_candidates, base_seed):
    metrics_path = os.path.join(reuse_sample_dir, "candidate_metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    source_rows = read_csv(metrics_path)
    if len(source_rows) < num_candidates:
        return None

    rows = []
    for source in sorted(source_rows, key=lambda item: to_int(item, "candidate_id"))[:num_candidates]:
        candidate_id = to_int(source, "candidate_id")
        seed = to_int(source, "seed", base_seed + candidate_id)
        src_dir = candidate_dir(reuse_sample_dir, candidate_id)
        dst_dir = candidate_dir(formal_sample_dir, candidate_id)
        copy_candidate(src_dir, dst_dir)

        row = OrderedDict()
        row["sample_id"] = source.get("sample_id") or os.path.basename(reuse_sample_dir)
        row["candidate_id"] = candidate_id
        row["seed"] = seed
        row.update(metrics_from_existing_row(source))
        rows.append(row)
    return rows


@torch.no_grad()
def generate_rows(sample, sample_index, net, args, formal_sample_dir):
    cond = sample["cond_image"].unsqueeze(0).to(args.device)
    cond01 = to_01(sample["cond_image"]).numpy()
    range01 = to_01(sample["range_mask"])[0].numpy()
    sample_id = sample_id_from_path(sample["path"])
    rows = []

    candidate_ids = list(range(args.num_candidates))
    for start in range(0, args.num_candidates, args.candidate_batch_size):
        chunk_ids = candidate_ids[start : start + args.candidate_batch_size]
        seeds = [args.base_seed + candidate_id for candidate_id in chunk_ids]
        cond_batch = cond.repeat(len(chunk_ids), 1, 1, 1)
        if args.seed_mode == "exact":
            outputs = restoration_with_candidate_seeds(net, cond_batch, seeds)
        else:
            outputs = restoration_with_fast_seed_batch(net, cond_batch, seeds)
        pred_batch = to_01(outputs).numpy()

        for local_idx, candidate_id in enumerate(chunk_ids):
            pred01 = pred_batch[local_idx, 0]
            dst_dir = candidate_dir(formal_sample_dir, candidate_id)
            save_candidate_images(dst_dir, pred01, range01, cond01)
            metrics, _ = metrics_from_prediction(pred01, range01, threshold=args.threshold)
            row = OrderedDict()
            row["sample_id"] = sample_id
            row["candidate_id"] = candidate_id
            row["seed"] = args.base_seed + candidate_id
            row.update(metrics)
            rows.append(row)

    return rows


def process_sample(sample, sample_index, args, reuse_root, module3_root, net):
    sample_id = sample_id_from_path(sample["path"])
    formal_sample_dir = ensure_dir(os.path.join(module3_root, sample_id))
    metrics_path = os.path.join(formal_sample_dir, "candidate_metrics.csv")

    if args.resume and os.path.exists(metrics_path):
        rows = read_csv(metrics_path)
        if len(rows) >= args.num_candidates and "strict_validator_score" in rows[0]:
            sorted_rows = sorted(rows, key=lambda item: (to_int(item, "strict_rank"), to_int(item, "candidate_id")))
            return sorted_rows

    rows = None
    if reuse_root:
        reuse_sample_dir = os.path.join(reuse_root, sample_id)
        if os.path.isdir(reuse_sample_dir):
            rows = load_reuse_rows(
                reuse_sample_dir,
                formal_sample_dir,
                args.num_candidates,
                args.base_seed,
            )

    if rows is None:
        if args.no_generate:
            return []
        rows = generate_rows(sample, sample_index, net, args, formal_sample_dir)

    rows = add_scores_and_ranks(rows, args.target_bcr, args.target_components)
    write_csv(metrics_path, rows)
    write_top_k(formal_sample_dir, rows, args.top_k)
    return rows


def mean(rows, key):
    values = [to_float(row, key) for row in rows]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def ratio(rows, key, predicate):
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if predicate(to_float(row, key))) / len(rows))


def build_global_outputs(output_dir, sample_rows, args):
    all_rows = []
    strict_top1 = []
    standard_top1 = []
    single_rows = []
    standard_vs_strict = []
    single_vs_validator = []

    for rows in sample_rows:
        if not rows:
            continue
        all_rows.extend(rows)
        strict_best = sorted(rows, key=lambda item: (to_int(item, "strict_rank"), to_int(item, "candidate_id")))[0]
        standard_best = sorted(rows, key=lambda item: (to_int(item, "standard_rank"), to_int(item, "candidate_id")))[0]
        single = sorted(rows, key=lambda item: to_int(item, "candidate_id"))[0]
        strict_top1.append(strict_best)
        standard_top1.append(standard_best)
        single_rows.append(single)

        standard_vs_strict.append(OrderedDict([
            ("sample_id", strict_best["sample_id"]),
            ("standard_top1_candidate_id", standard_best["candidate_id"]),
            ("strict_top1_candidate_id", strict_best["candidate_id"]),
            ("whether_top1_changed", str(standard_best["candidate_id"]) != str(strict_best["candidate_id"])),
            ("standard_top1_score", standard_best["standard_validator_score"]),
            ("strict_top1_score", strict_best["strict_validator_score"]),
            ("standard_top1_Pred_BCR", standard_best["Pred_BCR"]),
            ("strict_top1_Pred_BCR", strict_best["Pred_BCR"]),
            ("standard_top1_connected_components", standard_best["connected_components"]),
            ("strict_top1_connected_components", strict_best["connected_components"]),
            ("standard_top1_max_component_area_ratio", standard_best["max_component_area_ratio"]),
            ("strict_top1_max_component_area_ratio", strict_best["max_component_area_ratio"]),
            ("standard_top1_aspect_ratio_valid_ratio", standard_best["aspect_ratio_valid_ratio"]),
            ("strict_top1_aspect_ratio_valid_ratio", strict_best["aspect_ratio_valid_ratio"]),
        ]))

        single_vs_validator.append(OrderedDict([
            ("sample_id", strict_best["sample_id"]),
            ("single_candidate_id", single["candidate_id"]),
            ("validator_top1_candidate_id", strict_best["candidate_id"]),
            ("single_Pred_BCR", single["Pred_BCR"]),
            ("validator_Pred_BCR", strict_best["Pred_BCR"]),
            ("single_outside_violation", single["outside_violation"]),
            ("validator_outside_violation", strict_best["outside_violation"]),
            ("single_connected_components", single["connected_components"]),
            ("validator_connected_components", strict_best["connected_components"]),
            ("single_avg_component_area", single["avg_component_area"]),
            ("validator_avg_component_area", strict_best["avg_component_area"]),
            ("single_small_fragment_ratio", single["small_fragment_ratio"]),
            ("validator_small_fragment_ratio", strict_best["small_fragment_ratio"]),
            ("single_max_component_area_ratio", single["max_component_area_ratio"]),
            ("validator_max_component_area_ratio", strict_best["max_component_area_ratio"]),
            ("single_aspect_ratio_valid_ratio", single["aspect_ratio_valid_ratio"]),
            ("validator_aspect_ratio_valid_ratio", strict_best["aspect_ratio_valid_ratio"]),
            ("single_strict_score", single["strict_validator_score"]),
            ("validator_strict_score", strict_best["strict_validator_score"]),
        ]))

    all_path = os.path.join(output_dir, "all_candidates_metrics.csv")
    summary_path = os.path.join(output_dir, "validator_summary.csv")
    standard_vs_strict_path = os.path.join(output_dir, "standard_vs_strict_validator_comparison.csv")
    single_vs_validator_path = os.path.join(output_dir, "single_vs_validator_summary.csv")

    if all_rows:
        write_csv(all_path, all_rows)
        write_csv(standard_vs_strict_path, standard_vs_strict)
        write_csv(single_vs_validator_path, single_vs_validator)

    summary = OrderedDict([
        ("checkpoint_step", args.checkpoint_step),
        ("num_samples", len(strict_top1)),
        ("candidates_per_sample", args.num_candidates),
        ("top_k", args.top_k),
        ("target_bcr", args.target_bcr),
        ("target_components", args.target_components),
        ("threshold", args.threshold),
        ("avg_top1_Pred_BCR", mean(strict_top1, "Pred_BCR")),
        ("avg_top1_outside_violation", mean(strict_top1, "outside_violation")),
        ("avg_top1_connected_components", mean(strict_top1, "connected_components")),
        ("avg_top1_avg_component_area", mean(strict_top1, "avg_component_area")),
        ("avg_top1_small_fragment_ratio", mean(strict_top1, "small_fragment_ratio")),
        ("avg_top1_max_component_area_ratio", mean(strict_top1, "max_component_area_ratio")),
        ("avg_top1_aspect_ratio_mean", mean(strict_top1, "aspect_ratio_mean")),
        ("avg_top1_aspect_ratio_valid_ratio", mean(strict_top1, "aspect_ratio_valid_ratio")),
        ("avg_top1_standard_score", mean(strict_top1, "standard_validator_score")),
        ("avg_top1_strict_score", mean(strict_top1, "strict_validator_score")),
        ("all_candidates_avg_max_component_area_ratio", mean(all_rows, "max_component_area_ratio")),
        ("all_candidates_avg_aspect_ratio_valid_ratio", mean(all_rows, "aspect_ratio_valid_ratio")),
        ("single_avg_max_component_area_ratio", mean(single_rows, "max_component_area_ratio")),
        ("single_avg_aspect_ratio_valid_ratio", mean(single_rows, "aspect_ratio_valid_ratio")),
        ("max_component_area_ratio_over_0.50_ratio", ratio(strict_top1, "max_component_area_ratio", lambda v: v > 0.50)),
        ("max_component_area_ratio_over_0.65_ratio", ratio(strict_top1, "max_component_area_ratio", lambda v: v > 0.65)),
        ("aspect_ratio_valid_ratio_below_0.50_ratio", ratio(strict_top1, "aspect_ratio_valid_ratio", lambda v: v < 0.50)),
        ("aspect_ratio_valid_ratio_below_0.60_ratio", ratio(strict_top1, "aspect_ratio_valid_ratio", lambda v: v < 0.60)),
    ])
    write_csv(summary_path, [summary])
    return all_path, summary_path, standard_vs_strict_path, single_vs_validator_path, summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="experiments/module3_validator_checkpoint_90974")
    parser.add_argument("--reuse-candidate-root", default=None)
    parser.add_argument("--checkpoint-step", type=int, default=90974)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--seed-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument("--target-bcr", type=float, default=0.22)
    parser.add_argument("--target-components", type=float, default=4.0)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = load_json_with_comments(args.config)
    logger = StdLogger()
    dataset = build_dataset(config, logger)

    net = None
    if not args.no_generate:
        net, model_path = build_network(config, logger, args.device)
    else:
        model_path = "{}_Network.pth".format(config["path"]["resume_state"])

    output_dir = ensure_dir(args.output_dir)
    module3_root = ensure_dir(os.path.join(output_dir, "results", "module3_validator"))
    reuse_root = args.reuse_candidate_root
    if reuse_root:
        reuse_root = os.path.abspath(reuse_root)

    with open(os.path.join(output_dir, "module3_validator_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "module": "Module 3: Layout Candidate Validator",
            "checkpoint_step": args.checkpoint_step,
            "model_path": model_path,
            "num_candidates": args.num_candidates,
            "top_k": args.top_k,
            "base_seed": args.base_seed,
            "threshold": args.threshold,
            "target_bcr": args.target_bcr,
            "target_components": args.target_components,
            "standard_score": "2*abs(Pred_BCR-target_bcr)+abs(cc-target_components)/target_components+2*outside+fragment+max(0,max_ratio-0.50)+max(0,0.50-aspect_valid)",
            "strict_score": "2*abs(Pred_BCR-target_bcr)+abs(cc-target_components)/target_components+2*outside+fragment+1.5*max(0,max_ratio-0.45)+1.5*max(0,0.60-aspect_valid)",
            "reuse_candidate_root": reuse_root,
            "no_generate": args.no_generate,
        }, f, indent=2)

    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    print("Module 3 Layout Candidate Validator")
    print("checkpoint_step={}".format(args.checkpoint_step))
    print("output_dir={}".format(output_dir))
    print("samples={}, num_candidates={}, top_k={}".format(limit, args.num_candidates, args.top_k))
    print("reuse_candidate_root={}".format(reuse_root))
    print("generate_missing={}".format(not args.no_generate))

    sample_rows = []
    skipped = 0
    for index in tqdm(range(limit), desc="module3 samples"):
        sample = dataset[index]
        rows = process_sample(sample, index, args, reuse_root, module3_root, net)
        if not rows:
            skipped += 1
        sample_rows.append(rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paths = build_global_outputs(output_dir, sample_rows, args)
    print("processed_samples={}".format(paths[4]["num_samples"]))
    print("skipped_samples={}".format(skipped))
    print("all_candidates_metrics={}".format(paths[0]))
    print("validator_summary={}".format(paths[1]))
    print("standard_vs_strict_validator_comparison={}".format(paths[2]))
    print("single_vs_validator_summary={}".format(paths[3]))
    print("avg_top1_Pred_BCR={:.6f}".format(float(paths[4]["avg_top1_Pred_BCR"])))
    print("avg_top1_max_component_area_ratio={:.6f}".format(float(paths[4]["avg_top1_max_component_area_ratio"])))
    print("avg_top1_aspect_ratio_valid_ratio={:.6f}".format(float(paths[4]["avg_top1_aspect_ratio_valid_ratio"])))


if __name__ == "__main__":
    main()
