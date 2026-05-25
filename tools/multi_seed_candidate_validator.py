import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.praser import init_obj  # noqa: E402


class StdLogger:
    def info(self, message):
        print(message)

    def warning(self, message):
        print("WARNING: {}".format(message))


def load_json_with_comments(path):
    json_str = ""
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str, object_pairs_hook=OrderedDict)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_gray(array, path):
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def save_rgb(array, path):
    array = np.clip(array, 0.0, 1.0)
    Image.fromarray((array * 255.0).astype(np.uint8)).save(path)


def to_01(tensor):
    tensor = tensor.detach().float().cpu()
    if tensor.numel() > 0 and float(tensor.min()) < 0.0:
        tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0)


def make_generator(device, seed):
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def randn_per_candidate(shape, generators, device, dtype):
    chunks = [
        torch.randn(shape, generator=gen, device=device, dtype=dtype)
        for gen in generators
    ]
    return torch.stack(chunks, dim=0)


@torch.no_grad()
def restoration_with_candidate_seeds(net, y_cond, seeds):
    b, _, h, w = y_cond.shape
    out_channel = net.denoise_fn.out_channel
    device = y_cond.device
    dtype = y_cond.dtype
    generators = [make_generator(device, seed) for seed in seeds]
    y_t = randn_per_candidate((out_channel, h, w), generators, device, dtype)

    for i in reversed(range(0, net.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        model_mean, model_log_variance = net.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=True, y_cond=y_cond
        )
        if i > 0:
            noise = randn_per_candidate((out_channel, h, w), generators, device, dtype)
        else:
            noise = torch.zeros_like(y_t)
        y_t = model_mean + noise * (0.5 * model_log_variance).exp()

    if net.is_layout_condition(y_cond, y_t):
        y_t = net.apply_final_range_mask(y_t, y_cond)
    return y_t


@torch.no_grad()
def restoration_with_fast_seed_batch(net, y_cond, seeds):
    b, _, h, w = y_cond.shape
    out_channel = net.denoise_fn.out_channel
    device = y_cond.device
    dtype = y_cond.dtype

    init_generators = [make_generator(device, seed) for seed in seeds]
    y_t = randn_per_candidate((out_channel, h, w), init_generators, device, dtype)
    step_generator = make_generator(device, int(seeds[0]) + 1000003)

    for i in reversed(range(0, net.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        model_mean, model_log_variance = net.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=True, y_cond=y_cond
        )
        if i > 0:
            noise = torch.randn(
                y_t.shape, generator=step_generator, device=device, dtype=dtype
            )
        else:
            noise = torch.zeros_like(y_t)
        y_t = model_mean + noise * (0.5 * model_log_variance).exp()

    if net.is_layout_condition(y_cond, y_t):
        y_t = net.apply_final_range_mask(y_t, y_cond)
    return y_t


def component_measurements(binary, min_area=10, small_area=20):
    binary_u8 = binary.astype(np.uint8)
    total_area = float(binary_u8.sum())
    if total_area <= 0.0:
        return {
            "connected_components": 0,
            "avg_component_area": 0.0,
            "small_fragment_ratio": 0.0,
            "max_component_area_ratio": 0.0,
            "aspect_ratio_mean": 0.0,
            "aspect_ratio_valid_ratio": 0.0,
        }

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    valid_areas = []
    all_areas = []
    small_area_sum = 0.0
    aspect_ratios = []

    for label in range(1, num_labels):
        area = float(stats[label, cv2.CC_STAT_AREA])
        width = float(stats[label, cv2.CC_STAT_WIDTH])
        height = float(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= 0:
            continue
        all_areas.append(area)
        if area < small_area:
            small_area_sum += area
        if area >= min_area:
            valid_areas.append(area)
            short_side = max(min(width, height), 1.0)
            long_side = max(width, height)
            aspect_ratios.append(float(long_side / short_side))

    if not valid_areas:
        connected_components = 0
        avg_component_area = 0.0
    else:
        connected_components = len(valid_areas)
        avg_component_area = float(np.mean(valid_areas))

    if not all_areas:
        max_component_area_ratio = 0.0
    else:
        max_component_area_ratio = float(max(all_areas) / total_area)

    if not aspect_ratios:
        aspect_ratio_mean = 0.0
        aspect_ratio_valid_ratio = 0.0
    else:
        aspect_ratio_mean = float(np.mean(aspect_ratios))
        aspect_ratio_valid_ratio = float(
            sum(1 for ratio in aspect_ratios if ratio > 2.0) / len(aspect_ratios)
        )

    return {
        "connected_components": connected_components,
        "avg_component_area": avg_component_area,
        "small_fragment_ratio": float(small_area_sum / total_area),
        "max_component_area_ratio": max_component_area_ratio,
        "aspect_ratio_mean": aspect_ratio_mean,
        "aspect_ratio_valid_ratio": aspect_ratio_valid_ratio,
    }


def validator_score(metrics, args):
    pred_bcr = metrics["Pred_BCR"]
    if pred_bcr < args.bcr_min:
        bcr_penalty = args.bcr_min - pred_bcr
    elif pred_bcr > args.bcr_max:
        bcr_penalty = pred_bcr - args.bcr_max
    else:
        bcr_penalty = 0.0

    cc = metrics["connected_components"]
    component_penalty = 0.0 if args.component_min <= cc <= args.component_max else 1.0
    outside_penalty = metrics["outside_violation"]
    fragment_penalty = metrics["small_fragment_ratio"]

    max_ratio = metrics["max_component_area_ratio"]
    if max_ratio <= args.max_component_ratio_max:
        max_component_penalty = 0.0
    else:
        max_component_penalty = max_ratio - args.max_component_ratio_max

    aspect_valid = metrics["aspect_ratio_valid_ratio"]
    if aspect_valid >= args.aspect_ratio_valid_min:
        aspect_ratio_penalty = 0.0
    else:
        aspect_ratio_penalty = args.aspect_ratio_valid_min - aspect_valid

    score = (
        1.0 * bcr_penalty
        + 1.0 * component_penalty
        + 2.0 * outside_penalty
        + 0.5 * fragment_penalty
        + 0.5 * max_component_penalty
        + 0.5 * aspect_ratio_penalty
    )
    return {
        "bcr_penalty": bcr_penalty,
        "component_penalty": component_penalty,
        "outside_penalty": outside_penalty,
        "fragment_penalty": fragment_penalty,
        "max_component_penalty": max_component_penalty,
        "aspect_ratio_penalty": aspect_ratio_penalty,
        "validator_score": float(score),
    }


def candidate_metrics(pred01, range01, threshold, args):
    range_bin = (range01 > 0.5).astype(np.uint8)
    range_pixels = max(float(range_bin.sum()), 1.0)
    binary_raw = (pred01 > threshold).astype(np.uint8)
    binary_masked = (binary_raw * range_bin).astype(np.uint8)
    pred_total = float(binary_raw.sum())
    outside_white = float((binary_raw * (1 - range_bin)).sum())

    metrics = {
        "Pred_BCR": float(binary_masked.sum() / range_pixels),
        "outside_violation": 0.0 if pred_total == 0 else float(outside_white / pred_total),
    }
    metrics.update(
        component_measurements(
            binary_masked,
            min_area=args.component_min_area,
            small_area=args.small_fragment_area,
        )
    )
    metrics.update(validator_score(metrics, args))
    return metrics, binary_raw, binary_masked


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    rows = list(rows)
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def copy_top_candidate(src_dir, dst_dir):
    ensure_dir(dst_dir)
    for name in [
        "raw_output.png",
        "selected_binary.png",
        "binary_0.5.png",
        "binary_0.6.png",
        "binary_0.7.png",
        "condition_vis.png",
        "range_mask.png",
    ]:
        src = os.path.join(src_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, name))


def build_dataset(config, logger):
    dataset_opt = config["datasets"]["test"]["which_dataset"]
    return init_obj(dataset_opt, logger, default_file_name="data.dataset", init_type="Dataset")


def build_network(config, logger, device):
    network_opt = config["model"]["which_networks"][0]
    net = init_obj(network_opt, logger, default_file_name="models.network", init_type="Network")
    net.to(device)

    resume_state = config["path"]["resume_state"]
    if not resume_state:
        raise RuntimeError("config path.resume_state is required.")
    model_path = "{}_Network.pth".format(resume_state)
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state, strict=False)
    net.set_new_noise_schedule(device=device, phase="test")
    net.eval()
    return net, model_path


def save_candidate_images(candidate_dir, pred01, range01, cond01, binaries):
    ensure_dir(candidate_dir)
    save_gray(pred01, os.path.join(candidate_dir, "raw_output.png"))
    save_gray(range01, os.path.join(candidate_dir, "range_mask.png"))
    condition_rgb = np.transpose(cond01, (1, 2, 0))
    save_rgb(condition_rgb, os.path.join(candidate_dir, "condition_vis.png"))
    for key, binary in binaries.items():
        save_gray(binary.astype(np.float32), os.path.join(candidate_dir, "binary_{}.png".format(key)))
    shutil.copy2(
        os.path.join(candidate_dir, "binary_0.5.png"),
        os.path.join(candidate_dir, "selected_binary.png"),
    )


def process_sample(sample, sample_index, net, args, output_root, all_rows):
    sample_path = sample["path"]
    sample_id = os.path.splitext(os.path.basename(sample_path))[0]
    sample_dir = ensure_dir(os.path.join(output_root, "candidate_sampling", sample_id))
    metrics_path = os.path.join(sample_dir, "candidate_metrics.csv")

    if args.resume and os.path.exists(metrics_path):
        rows = list(csv.DictReader(open(metrics_path, "r", encoding="utf-8")))
        if len(rows) >= args.candidates:
            all_rows.extend(rows)
            return rows

    cond = sample["cond_image"].unsqueeze(0).to(args.device)
    cond01 = to_01(sample["cond_image"]).numpy()
    range01 = to_01(sample["range_mask"])[0].numpy()

    rows = []
    candidate_ids = list(range(args.candidates))
    for start in range(0, args.candidates, args.candidate_batch_size):
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
            threshold_binaries = {}
            metrics = None
            for threshold in args.thresholds:
                key = "{:.1f}".format(threshold)
                threshold_metrics, _, binary_masked = candidate_metrics(
                    pred01, range01, threshold, args
                )
                threshold_binaries[key] = binary_masked
                if abs(threshold - args.selected_threshold) < 1e-8:
                    metrics = threshold_metrics
            candidate_dir = ensure_dir(
                os.path.join(sample_dir, "candidate_{:02d}".format(candidate_id))
            )
            save_candidate_images(candidate_dir, pred01, range01, cond01, threshold_binaries)
            row = OrderedDict(
                [
                    ("sample_id", sample_id),
                    ("sample_index", sample_index),
                    ("candidate_id", candidate_id),
                    ("seed", args.base_seed + candidate_id),
                    ("Pred_BCR", metrics["Pred_BCR"]),
                    ("outside_violation", metrics["outside_violation"]),
                    ("connected_components", metrics["connected_components"]),
                    ("avg_component_area", metrics["avg_component_area"]),
                    ("small_fragment_ratio", metrics["small_fragment_ratio"]),
                    ("max_component_area_ratio", metrics["max_component_area_ratio"]),
                    ("aspect_ratio_mean", metrics["aspect_ratio_mean"]),
                    ("aspect_ratio_valid_ratio", metrics["aspect_ratio_valid_ratio"]),
                    ("bcr_penalty", metrics["bcr_penalty"]),
                    ("component_penalty", metrics["component_penalty"]),
                    ("outside_penalty", metrics["outside_penalty"]),
                    ("fragment_penalty", metrics["fragment_penalty"]),
                    ("max_component_penalty", metrics["max_component_penalty"]),
                    ("aspect_ratio_penalty", metrics["aspect_ratio_penalty"]),
                    ("validator_score", metrics["validator_score"]),
                    ("rank", 0),
                    ("candidate_dir", candidate_dir),
                ]
            )
            rows.append(row)

    rows.sort(key=lambda row: (float(row["validator_score"]), int(row["candidate_id"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    top_root = ensure_dir(os.path.join(sample_dir, "top_k_candidates"))
    for row in rows[: args.top_k]:
        src_dir = row["candidate_dir"]
        dst_dir = os.path.join(
            top_root,
            "rank_{:02d}_candidate_{:02d}".format(int(row["rank"]), int(row["candidate_id"])),
        )
        copy_top_candidate(src_dir, dst_dir)

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(metrics_path, rows, fieldnames)
    all_rows.extend(rows)
    return rows


def build_summaries(output_root, sample_rows, all_rows, args, checkpoint_step, model_path):
    all_fieldnames = list(all_rows[0].keys()) if all_rows else []
    all_path = os.path.join(output_root, "all_candidates_metrics.csv")
    write_csv(all_path, all_rows, all_fieldnames)

    best_rows = [rows[0] for rows in sample_rows if rows]
    summary = OrderedDict(
        [
            ("checkpoint_step", checkpoint_step),
            ("model_path", model_path),
            ("num_samples", len(best_rows)),
            ("candidates_per_sample", args.candidates),
            ("top_k", args.top_k),
            ("selected_threshold", args.selected_threshold),
            ("avg_best_score", mean(best_rows, "validator_score")),
            ("avg_best_Pred_BCR", mean(best_rows, "Pred_BCR")),
            ("avg_best_outside_violation", mean(best_rows, "outside_violation")),
            ("avg_best_connected_components", mean(best_rows, "connected_components")),
            ("avg_best_avg_component_area", mean(best_rows, "avg_component_area")),
            ("avg_best_small_fragment_ratio", mean(best_rows, "small_fragment_ratio")),
            ("avg_best_max_component_area_ratio", mean(best_rows, "max_component_area_ratio")),
            ("avg_best_aspect_ratio_valid_ratio", mean(best_rows, "aspect_ratio_valid_ratio")),
        ]
    )
    summary_path = os.path.join(output_root, "candidate_validator_summary.csv")
    write_csv(summary_path, [summary], list(summary.keys()))

    per_sample_rows = []
    metric_keys = [
        "Pred_BCR",
        "outside_violation",
        "connected_components",
        "avg_component_area",
        "small_fragment_ratio",
        "max_component_area_ratio",
        "aspect_ratio_valid_ratio",
        "validator_score",
    ]
    for rows in sample_rows:
        if not rows:
            continue
        single = sorted(rows, key=lambda row: int(row["candidate_id"]))[0]
        best = rows[0]
        row = OrderedDict(
            [
                ("sample_id", best["sample_id"]),
                ("single_candidate_id", single["candidate_id"]),
                ("single_seed", single["seed"]),
                ("top1_candidate_id", best["candidate_id"]),
                ("top1_seed", best["seed"]),
            ]
        )
        for key in metric_keys:
            row["single_{}".format(key)] = single[key]
            row["top1_{}".format(key)] = best[key]
            row["delta_{}".format(key)] = float(best[key]) - float(single[key])
        per_sample_rows.append(row)

    per_sample_path = os.path.join(output_root, "single_vs_validator_per_sample.csv")
    write_csv(per_sample_path, per_sample_rows, list(per_sample_rows[0].keys()))

    single_rows = []
    top1_rows = []
    for rows in sample_rows:
        if not rows:
            continue
        single_rows.append(sorted(rows, key=lambda row: int(row["candidate_id"]))[0])
        top1_rows.append(rows[0])
    compare = OrderedDict(
        [
            ("checkpoint_step", checkpoint_step),
            ("num_samples", len(top1_rows)),
            ("candidates_per_sample", args.candidates),
            ("top_k", args.top_k),
        ]
    )
    for key in metric_keys:
        compare["single_avg_{}".format(key)] = mean(single_rows, key)
        compare["top1_avg_{}".format(key)] = mean(top1_rows, key)
        compare["delta_avg_{}".format(key)] = (
            compare["top1_avg_{}".format(key)] - compare["single_avg_{}".format(key)]
        )

    compare_path = os.path.join(output_root, "single_vs_validator_summary.csv")
    write_csv(compare_path, [compare], list(compare.keys()))
    return all_path, summary_path, compare_path, per_sample_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--checkpoint-step", type=int, default=90974)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--seed-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--candidate-batch-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--selected-threshold", type=float, default=0.5)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    parser.add_argument("--bcr-min", type=float, default=0.10)
    parser.add_argument("--bcr-max", type=float, default=0.35)
    parser.add_argument("--component-min", type=int, default=2)
    parser.add_argument("--component-max", type=int, default=12)
    parser.add_argument("--component-min-area", type=int, default=10)
    parser.add_argument("--small-fragment-area", type=int, default=20)
    parser.add_argument("--max-component-ratio-max", type=float, default=0.65)
    parser.add_argument("--aspect-ratio-valid-min", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    config = load_json_with_comments(args.config)
    logger = StdLogger()
    dataset = build_dataset(config, logger)
    net, model_path = build_network(config, logger, device)

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_root = args.output_dir or os.path.join(
        "experiments",
        "candidate_validator_checkpoint_{}_{}".format(args.checkpoint_step, timestamp),
    )
    output_root = ensure_dir(output_root)

    with open(os.path.join(output_root, "candidate_validator_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_step": args.checkpoint_step,
                "model_path": model_path,
                "base_seed": args.base_seed,
                "seed_mode": args.seed_mode,
                "candidates": args.candidates,
                "candidate_batch_size": args.candidate_batch_size,
                "top_k": args.top_k,
                "selected_threshold": args.selected_threshold,
                "thresholds": args.thresholds,
                "validator": {
                    "bcr_min": args.bcr_min,
                    "bcr_max": args.bcr_max,
                    "component_min": args.component_min,
                    "component_max": args.component_max,
                    "avg_component_area_min": 50,
                    "avg_component_area_max": 4000,
                    "small_fragment_ratio_max": 0.15,
                    "max_component_ratio_max": args.max_component_ratio_max,
                    "aspect_ratio_valid_min": args.aspect_ratio_valid_min,
                },
            },
            f,
            indent=2,
        )

    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    print("Using checkpoint_step={} model={}".format(args.checkpoint_step, model_path))
    print("Dataset samples={}, running samples={}, candidates_per_sample={}".format(len(dataset), limit, args.candidates))
    print("Output root={}".format(output_root))

    all_rows = []
    sample_rows = []
    for index in tqdm(range(limit), desc="candidate validator samples"):
        sample = dataset[index]
        rows = process_sample(sample, index, net, args, output_root, all_rows)
        sample_rows.append(rows)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paths = build_summaries(output_root, sample_rows, all_rows, args, args.checkpoint_step, model_path)
    print("Wrote all_candidates_metrics={}".format(paths[0]))
    print("Wrote candidate_validator_summary={}".format(paths[1]))
    print("Wrote single_vs_validator_summary={}".format(paths[2]))
    print("Wrote single_vs_validator_per_sample={}".format(paths[3]))


if __name__ == "__main__":
    main()
