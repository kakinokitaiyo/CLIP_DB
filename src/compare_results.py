from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = ROOT / "ground_truth.json"
DEFAULT_OUTPUTS = ROOT / "outputs"

CONDITION_DIRS = {
    "clip_photos": ROOT / "outputs" / "clip_photos",
    "clip_rbte": ROOT / "outputs" / "clip_rbte",
    "sbir_photos": ROOT / "outputs" / "sbir_photos",
    "sbir_rbte": ROOT / "outputs" / "sbir_rbte",
}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ground_truth(gt_raw: Dict[str, Any]) -> Dict[str, List[str]]:
    gt: Dict[str, List[str]] = {}
    for sketch_file, value in gt_raw.items():
        if isinstance(value, str):
            gt[sketch_file] = [value]
        elif isinstance(value, list):
            gt[sketch_file] = value
        else:
            raise ValueError(f"ground_truth の値は str か list にしてください: {sketch_file}")
    return gt


def load_summary(summary_path: Path) -> Dict[str, List[str]]:
    """
    summary.json を
    {
      "writing_1.png": ["apple.jpeg", "banana.jpeg", ...],
      ...
    }
    の形に変換する
    """
    raw = load_json(summary_path)
    result: Dict[str, List[str]] = {}

    for item in raw:
        sketch_file = item["sketch_file"]
        ranked_files = [x["gallery_file"] for x in item["top5"]]
        result[sketch_file] = ranked_files

    return result


def get_best_rank(preds: List[str], gt_list: List[str]) -> int | None:
    """
    gt候補が複数ある場合、その中で最も良い順位を返す
    1始まり。見つからなければ None
    """
    gt_set = set(gt_list)
    for i, pred in enumerate(preds, start=1):
        if pred in gt_set:
            return i
    return None


def evaluate_condition(
    gt: Dict[str, List[str]],
    preds: Dict[str, List[str]],
    topk: int = 5,
) -> Dict[str, Any]:
    rows = []
    total = 0
    top1_correct = 0
    top5_correct = 0
    rank_sum = 0
    rank_count = 0

    for sketch_file, gt_list in gt.items():
        total += 1
        ranked = preds.get(sketch_file, [])
        best_rank = get_best_rank(ranked, gt_list)

        is_top1 = best_rank == 1
        is_top5 = best_rank is not None and best_rank <= topk

        if is_top1:
            top1_correct += 1
        if is_top5:
            top5_correct += 1
        if best_rank is not None:
            rank_sum += best_rank
            rank_count += 1

        rows.append(
            {
                "sketch_file": sketch_file,
                "ground_truth": ", ".join(gt_list),
                "pred_1": ranked[0] if len(ranked) >= 1 else "",
                "pred_2": ranked[1] if len(ranked) >= 2 else "",
                "pred_3": ranked[2] if len(ranked) >= 3 else "",
                "pred_4": ranked[3] if len(ranked) >= 4 else "",
                "pred_5": ranked[4] if len(ranked) >= 5 else "",
                "best_rank": best_rank if best_rank is not None else "",
                "top1_correct": is_top1,
                "top5_correct": is_top5,
            }
        )

    summary = {
        "total": total,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "top5_accuracy": top5_correct / total if total else 0.0,
        "mean_rank_found_only": rank_sum / rank_count if rank_count else None,
        "rows": rows,
    }
    return summary


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_comparison_table(
    gt: Dict[str, List[str]],
    condition_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    table = []

    for sketch_file, gt_list in gt.items():
        row: Dict[str, Any] = {
            "sketch_file": sketch_file,
            "ground_truth": ", ".join(gt_list),
        }

        for cond_name, result in condition_results.items():
            row_map = {r["sketch_file"]: r for r in result["rows"]}
            r = row_map[sketch_file]
            row[f"{cond_name}_pred1"] = r["pred_1"]
            row[f"{cond_name}_best_rank"] = r["best_rank"]
            row[f"{cond_name}_top1"] = r["top1_correct"]
            row[f"{cond_name}_top5"] = r["top5_correct"]

        table.append(row)

    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="4条件の summary.json を比較して Top1/Top5 を集計する")
    parser.add_argument("--ground_truth", type=Path, default=DEFAULT_GT, help="ground_truth.json")
    parser.add_argument("--topk", type=int, default=5, help="Top-k の k")
    parser.add_argument("--save_dir", type=Path, default=ROOT / "outputs" / "comparison", help="保存先")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gt_path = args.ground_truth.expanduser().resolve()
    save_dir = args.save_dir.expanduser().resolve()
    topk = args.topk

    if not gt_path.is_file():
        raise FileNotFoundError(f"ground_truth.json が見つかりません: {gt_path}")

    gt = normalize_ground_truth(load_json(gt_path))

    save_dir.mkdir(parents=True, exist_ok=True)

    condition_results: Dict[str, Dict[str, Any]] = {}

    print("\n=== Overall Results ===")
    for cond_name, cond_dir in CONDITION_DIRS.items():
        summary_path = cond_dir / "summary.json"
        if not summary_path.is_file():
            print(f"[skip] {cond_name}: summary.json がありません -> {summary_path}")
            continue

        preds = load_summary(summary_path)
        result = evaluate_condition(gt, preds, topk=topk)
        condition_results[cond_name] = result

        print(f"\n[{cond_name}]")
        print(f"  total         : {result['total']}")
        print(f"  top1_correct  : {result['top1_correct']}")
        print(f"  top5_correct  : {result['top5_correct']}")
        print(f"  top1_accuracy : {result['top1_accuracy']:.3f}")
        print(f"  top5_accuracy : {result['top5_accuracy']:.3f}")
        if result["mean_rank_found_only"] is None:
            print(f"  mean_rank     : N/A")
        else:
            print(f"  mean_rank     : {result['mean_rank_found_only']:.3f}")

        out_json = save_dir / f"{cond_name}_evaluation.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        out_csv = save_dir / f"{cond_name}_evaluation.csv"
        save_csv(out_csv, result["rows"])

    comparison_table = build_comparison_table(gt, condition_results)
    comparison_csv = save_dir / "comparison_table.csv"
    save_csv(comparison_csv, comparison_table)

    comparison_json = save_dir / "comparison_table.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(comparison_table, f, ensure_ascii=False, indent=2)

    overall_summary = {}
    for cond_name, result in condition_results.items():
        overall_summary[cond_name] = {
            "total": result["total"],
            "top1_correct": result["top1_correct"],
            "top5_correct": result["top5_correct"],
            "top1_accuracy": result["top1_accuracy"],
            "top5_accuracy": result["top5_accuracy"],
            "mean_rank_found_only": result["mean_rank_found_only"],
        }

    overall_json = save_dir / "overall_summary.json"
    with open(overall_json, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"\nsaved to: {save_dir}")


if __name__ == "__main__":
    main()