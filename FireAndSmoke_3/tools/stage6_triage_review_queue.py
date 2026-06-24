from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


CASE_WEIGHT = {
    "empty_with_detection": 100,
    "false_positive": 90,
    "mixed_fp_miss": 80,
    "missed_or_incomplete": 70,
    "video_detection_needs_review": 45,
    "video_empty_sample": 5,
}


def load_rows(run_name: str, manifest: Path) -> list[dict[str, Any]]:
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["run"] = run_name
        rows.append(row)
    return rows


def row_key(row: dict[str, Any]) -> str:
    return f'{row.get("source", "")}|{row.get("frame_index", "")}'


def max_conf(row: dict[str, Any]) -> float:
    preds = row.get("predictions", []) or []
    if not preds:
        return 0.0
    return max(float(pred.get("confidence", 0.0)) for pred in preds)


def score_group(rows: list[dict[str, Any]]) -> tuple[float, str]:
    models = {row["run"] for row in rows}
    case_types = [str(row.get("case_type", "")) for row in rows]
    base = max(CASE_WEIGHT.get(case, 10) for case in case_types)
    consensus_bonus = 20 * (len(models) - 1)
    conf_bonus = 20 * max(max_conf(row) for row in rows)
    fp_bonus = max(float(row.get("false_positive_count") or 0) for row in rows) * 2
    miss_bonus = max(float(row.get("missed_count") or 0) for row in rows)
    label_count = max(float(row.get("label_count") or 0) for row in rows if str(row.get("label_count", "")) != "" or True)
    if "video_detection_needs_review" in case_types and label_count == 0:
        reason = "video_unlabeled_consensus" if len(models) > 1 else "video_unlabeled_single_model"
    elif "empty_with_detection" in case_types:
        reason = "safe_empty_label_false_positive"
    elif "false_positive" in case_types:
        reason = "labeled_image_false_positive"
    elif "mixed_fp_miss" in case_types:
        reason = "labeled_image_mixed_false_positive_and_miss"
    elif "missed_or_incomplete" in case_types:
        reason = "labeled_image_missed_or_incomplete"
    else:
        reason = "other"
    return base + consensus_bonus + conf_bonus + fp_bonus + miss_bonus, reason


def copy_case(rows: list[dict[str, Any]], out_dir: Path, rank: int, reason: str) -> tuple[str, str]:
    best = max(rows, key=max_conf)
    prefix = f"{rank:04d}_{reason}_{Path(str(best.get('source', 'case'))).stem}"
    annotated_src = Path(str(best.get("annotated_image", "")))
    review_src = Path(str(best.get("review_image", "")))
    annotated_dst = out_dir / "top_annotated" / f"{prefix}.jpg"
    review_dst = out_dir / "top_review_images" / f"{prefix}.jpg"
    annotated_dst.parent.mkdir(parents=True, exist_ok=True)
    review_dst.parent.mkdir(parents=True, exist_ok=True)
    if annotated_src.exists():
        shutil.copy2(annotated_src, annotated_dst)
    if review_src.exists():
        shutil.copy2(review_src, review_dst)
    return str(annotated_dst), str(review_dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress Stage6 multi-model mining outputs into a prioritized review queue.")
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "MANIFEST_JSONL"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    for run_name, manifest_value in args.run:
        rows = load_rows(run_name, Path(manifest_value))
        total_rows += len(rows)
        for row in rows:
            grouped[row_key(row)].append(row)

    scored = []
    for key, rows in grouped.items():
        score, reason = score_group(rows)
        scored.append((score, reason, key, rows))
    scored.sort(key=lambda item: item[0], reverse=True)

    args.out.mkdir(parents=True, exist_ok=True)
    compact_rows = []
    for rank, (score, reason, key, rows) in enumerate(scored[: args.top_k], start=1):
        annotated, review = copy_case(rows, args.out, rank, reason)
        models = sorted({row["run"] for row in rows})
        case_types = sorted({str(row.get("case_type", "")) for row in rows})
        best = max(rows, key=max_conf)
        compact_rows.append(
            {
                "rank": rank,
                "score": round(score, 3),
                "reason": reason,
                "models": ";".join(models),
                "case_types": ";".join(case_types),
                "source": best.get("source", ""),
                "frame_index": best.get("frame_index", ""),
                "source_type": best.get("source_type", ""),
                "label_count": best.get("label_count", ""),
                "prediction_count_max": max(int(row.get("prediction_count") or 0) for row in rows),
                "false_positive_count_max": max(int(row.get("false_positive_count") or 0) for row in rows if str(row.get("false_positive_count", "")) != "" or True),
                "missed_count_max": max(int(row.get("missed_count") or 0) for row in rows if str(row.get("missed_count", "")) != "" or True),
                "max_confidence": round(max(max_conf(row) for row in rows), 5),
                "annotated_image": annotated,
                "review_image": review,
                "review_decision": "pending",
                "notes": "",
            }
        )

    with (args.out / "prioritized_review_queue.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(compact_rows[0].keys()) if compact_rows else ["rank"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact_rows)

    summary = {
        "input_rows": total_rows,
        "unique_cases": len(scored),
        "top_k": args.top_k,
        "queue_csv": str(args.out / "prioritized_review_queue.csv"),
        "auto_strategy": [
            "Use verified-empty mining for zero-manual hard negatives.",
            "Use this queue only for high-value ambiguous cases, not full labeling.",
        ],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
