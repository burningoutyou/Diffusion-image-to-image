import argparse
import csv
import os
import shutil
from collections import OrderedDict
from pathlib import Path


IMAGE_NAMES = [
    "raw_output.png",
    "selected_binary.png",
    "binary_0.5.png",
    "binary_0.6.png",
    "binary_0.7.png",
    "condition_vis.png",
    "range_mask.png",
]


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


def new_score(row, target_bcr, target_components):
    pred_bcr = to_float(row, "Pred_BCR")
    connected_components = to_float(row, "connected_components")
    outside_violation = to_float(row, "outside_violation")
    small_fragment_ratio = to_float(row, "small_fragment_ratio")
    max_component_area_ratio = to_float(row, "max_component_area_ratio")
    aspect_ratio_valid_ratio = to_float(row, "aspect_ratio_valid_ratio")

    return (
        2.0 * abs(pred_bcr - target_bcr)
        + 1.0 * abs(connected_components - target_components) / float(target_components)
        + 2.0 * outside_violation
        + 1.0 * small_fragment_ratio
        + 1.0 * max(0.0, max_component_area_ratio - 0.50)
        + 1.0 * max(0.0, 0.50 - aspect_ratio_valid_ratio)
    )


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_candidate_metrics(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("candidate_metrics.csv"))


def normalized_candidate_id(row):
    return to_int(row, "candidate_id")


def infer_candidate_dir(metrics_path, row):
    candidate_dir = row.get("candidate_dir")
    if candidate_dir and os.path.isdir(candidate_dir):
        return candidate_dir
    candidate_id = normalized_candidate_id(row)
    sibling = metrics_path.parent / "candidate_{:02d}".format(candidate_id)
    if sibling.is_dir():
        return str(sibling)
    return None


def copy_top_k(metrics_path, rows, top_k):
    top_dir = metrics_path.parent / "top_k_candidates_rescored"
    top_dir.mkdir(parents=True, exist_ok=True)

    for row in rows[:top_k]:
        src_dir = infer_candidate_dir(metrics_path, row)
        if src_dir is None:
            continue
        rank = to_int(row, "rank")
        candidate_id = normalized_candidate_id(row)
        dst_dir = top_dir / "rank_{:02d}_candidate_{:02d}".format(rank, candidate_id)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in IMAGE_NAMES:
            src = Path(src_dir) / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)


def prepare_rows(rows, target_bcr, target_components):
    prepared = []
    for row in rows:
        out = OrderedDict(row)
        out["old_validator_score"] = row.get("old_validator_score", row.get("validator_score", ""))
        out["old_rank"] = row.get("old_rank", row.get("rank", ""))
        out["new_validator_score"] = new_score(row, target_bcr, target_components)
        out["new_rank"] = 0
        prepared.append(out)

    prepared.sort(key=lambda item: (to_float(item, "new_validator_score"), normalized_candidate_id(item)))
    for index, row in enumerate(prepared, start=1):
        row["new_rank"] = index
        row["validator_score"] = row["new_validator_score"]
        row["rank"] = index
    return prepared


def fieldnames_for(rows):
    ordered = []
    for row in rows:
        for key in row.keys():
            if key not in ordered:
                ordered.append(key)
    return ordered


def old_top1(rows):
    return sorted(
        rows,
        key=lambda row: (
            to_int(row, "old_rank", 999999),
            to_float(row, "old_validator_score", 999999.0),
            normalized_candidate_id(row),
        ),
    )[0]


def sample_summary(sample_id, original_rows, rescored_rows):
    old = old_top1(rescored_rows)
    new = rescored_rows[0]
    return OrderedDict(
        [
            ("sample_id", sample_id),
            ("num_candidates", len(rescored_rows)),
            ("old_top1_candidate_id", old.get("candidate_id", "")),
            ("old_top1_score", old.get("old_validator_score", "")),
            ("new_top1_candidate_id", new.get("candidate_id", "")),
            ("new_top1_score", new.get("new_validator_score", "")),
            ("new_top1_Pred_BCR", new.get("Pred_BCR", "")),
            ("new_top1_outside_violation", new.get("outside_violation", "")),
            ("new_top1_connected_components", new.get("connected_components", "")),
            ("new_top1_avg_component_area", new.get("avg_component_area", "")),
            ("new_top1_small_fragment_ratio", new.get("small_fragment_ratio", "")),
            ("new_top1_max_component_area_ratio", new.get("max_component_area_ratio", "")),
            ("new_top1_aspect_ratio_valid_ratio", new.get("aspect_ratio_valid_ratio", "")),
        ]
    )


def comparison_row(sample_id, rescored_rows):
    old = old_top1(rescored_rows)
    new = rescored_rows[0]
    changed = str(old.get("candidate_id", "")) != str(new.get("candidate_id", ""))
    return OrderedDict(
        [
            ("sample_id", sample_id),
            ("old_top1_candidate_id", old.get("candidate_id", "")),
            ("new_top1_candidate_id", new.get("candidate_id", "")),
            ("whether_top1_changed", changed),
            ("old_top1_score", old.get("old_validator_score", "")),
            ("new_top1_score", new.get("new_validator_score", "")),
            ("old_top1_Pred_BCR", old.get("Pred_BCR", "")),
            ("new_top1_Pred_BCR", new.get("Pred_BCR", "")),
            ("old_top1_connected_components", old.get("connected_components", "")),
            ("new_top1_connected_components", new.get("connected_components", "")),
            ("old_top1_max_component_area_ratio", old.get("max_component_area_ratio", "")),
            ("new_top1_max_component_area_ratio", new.get("max_component_area_ratio", "")),
            ("old_top1_aspect_ratio_valid_ratio", old.get("aspect_ratio_valid_ratio", "")),
            ("new_top1_aspect_ratio_valid_ratio", new.get("aspect_ratio_valid_ratio", "")),
        ]
    )


def mean(rows, key):
    values = [to_float(row, key) for row in rows]
    if not values:
        return 0.0
    return sum(values) / len(values)


def choose_output_root(input_path):
    path = Path(input_path)
    if path.is_file():
        return path.parent
    return path


def process_file(metrics_path, args):
    rows = read_csv(metrics_path)
    if not rows:
        return None, None, []

    sample_id = rows[0].get("sample_id") or metrics_path.parent.name
    rescored = prepare_rows(rows, args.target_bcr, args.target_components)
    output_path = metrics_path.with_name("candidate_metrics_rescored.csv")
    write_csv(str(output_path), rescored, fieldnames_for(rescored))
    copy_top_k(metrics_path, rescored, args.top_k)
    return sample_summary(sample_id, rows, rescored), comparison_row(sample_id, rescored), rescored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="candidate_metrics.csv or directory containing candidate_metrics.csv files")
    parser.add_argument("--target-bcr", type=float, default=0.22)
    parser.add_argument("--target-components", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    metrics_files = find_candidate_metrics(args.input)
    output_root = choose_output_root(args.input)
    summaries = []
    comparisons = []
    all_rescored = []

    for metrics_path in metrics_files:
        summary, comparison, rescored = process_file(metrics_path, args)
        if summary is None:
            continue
        summaries.append(summary)
        comparisons.append(comparison)
        all_rescored.extend(rescored)

    summary_path = output_root / "rescored_validator_summary.csv"
    comparison_path = output_root / "old_vs_new_validator_comparison.csv"
    all_path = output_root / "all_candidates_metrics_rescored.csv"

    if summaries:
        write_csv(str(summary_path), summaries, fieldnames_for(summaries))
        write_csv(str(comparison_path), comparisons, fieldnames_for(comparisons))
        write_csv(str(all_path), all_rescored, fieldnames_for(all_rescored))

    changed_count = sum(1 for row in comparisons if str(row["whether_top1_changed"]) == "True")
    avg_candidates = mean(summaries, "num_candidates")
    changed_ratio = 0.0 if not summaries else changed_count / len(summaries)

    print("Processed candidate_metrics.csv files: {}".format(len(summaries)))
    print("Average candidates per sample: {:.4f}".format(avg_candidates))
    print("Top1 changed: {} / {} ({:.2%})".format(changed_count, len(summaries), changed_ratio))
    print("New top1 avg Pred_BCR: {:.6f}".format(mean(summaries, "new_top1_Pred_BCR")))
    print("New top1 avg connected_components: {:.6f}".format(mean(summaries, "new_top1_connected_components")))
    print("New top1 avg max_component_area_ratio: {:.6f}".format(mean(summaries, "new_top1_max_component_area_ratio")))
    print("New top1 avg aspect_ratio_valid_ratio: {:.6f}".format(mean(summaries, "new_top1_aspect_ratio_valid_ratio")))
    print("rescored_validator_summary: {}".format(summary_path))
    print("old_vs_new_validator_comparison: {}".format(comparison_path))
    print("all_candidates_metrics_rescored: {}".format(all_path))


if __name__ == "__main__":
    main()
