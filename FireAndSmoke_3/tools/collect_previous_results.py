from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]


def read_final_metrics(results_csv: Path) -> dict[str, str]:
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in reader]
    if not rows:
        return {}
    last = rows[-1]
    out = {"run": str(results_csv.parent), "epochs": last.get("epoch", str(len(rows)))}
    for col in METRIC_COLUMNS:
        value = last.get(col, "")
        out[col] = value.strip() if isinstance(value, str) else value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect prior YOLO training metrics.")
    parser.add_argument("--roots", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for root in args.roots:
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("results.csv")):
            metrics = read_final_metrics(csv_path)
            if metrics:
                metrics["name"] = csv_path.parent.name
                rows.append(metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "run", "epochs"] + METRIC_COLUMNS
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} runs -> {args.out}")


if __name__ == "__main__":
    main()
