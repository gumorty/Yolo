from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_audit import audit_dataset


def count_class_size(audit: dict[str, Any], class_name: str, size_prefixes: tuple[str, ...], split: str | None = None) -> int:
    if split:
        bins = audit["splits"].get(split, {}).get("class_area_bins", {})
    else:
        bins = audit["totals"].get("class_area_bins", {})
    total = 0
    for key, value in bins.items():
        name, _, size = key.partition("|")
        if name == class_name and any(size.startswith(prefix) for prefix in size_prefixes):
            total += int(value)
    return total


def highres_ratio(audit: dict[str, Any]) -> float:
    shapes = audit["totals"].get("image_shapes", {})
    total = sum(int(v) for v in shapes.values())
    if not total:
        return 0.0
    low = int(shapes.get("max_side<=640", 0))
    return round((total - low) / total, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a Stage4 YOLO dataset is ready for scout training.")
    parser.add_argument("--data", type=Path, help="Dataset data.yaml. If provided, an audit is generated first.")
    parser.add_argument("--audit", type=Path, help="Existing audit JSON.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-train-small-fire", type=int, default=5000)
    parser.add_argument("--min-train-small-smoke", type=int, default=1000)
    parser.add_argument("--min-val-small-fire", type=int, default=300)
    parser.add_argument("--min-val-small-smoke", type=int, default=100)
    parser.add_argument("--min-highres-ratio", type=float, default=0.20)
    parser.add_argument("--min-other-small", type=int, default=1000)
    args = parser.parse_args()

    if args.data:
        audit = audit_dataset(args.data)
    elif args.audit:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
    else:
        raise SystemExit("Pass --data or --audit.")

    checks = []

    def add_check(name: str, value: int | float, threshold: int | float, op: str = ">=") -> None:
        passed = value >= threshold if op == ">=" else value <= threshold
        checks.append({"name": name, "value": value, "threshold": threshold, "op": op, "pass": passed})

    add_check("train tiny+small fire", count_class_size(audit, "fire", ("tiny", "small"), "train"), args.min_train_small_fire)
    add_check("train tiny+small smoke", count_class_size(audit, "smoke", ("tiny", "small"), "train"), args.min_train_small_smoke)
    val_split = "valid" if "valid" in audit.get("splits", {}) else "val"
    add_check(f"{val_split} tiny+small fire", count_class_size(audit, "fire", ("tiny", "small"), val_split), args.min_val_small_fire)
    add_check(f"{val_split} tiny+small smoke", count_class_size(audit, "smoke", ("tiny", "small"), val_split), args.min_val_small_smoke)
    add_check("total tiny+small other hard negatives", count_class_size(audit, "other", ("tiny", "small")), args.min_other_small)
    add_check("high-resolution source ratio", highres_ratio(audit), args.min_highres_ratio)

    report = {
        "data_yaml": audit.get("data_yaml"),
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "image_shapes": audit["totals"].get("image_shapes", {}),
        "recommendation": "",
    }
    if report["pass"]:
        report["recommendation"] = "Dataset prior is acceptable for 10-20 epoch scout training. Do not start 120 epochs until scout improves the fixed holdout metrics."
    else:
        failed = [item["name"] for item in checks if not item["pass"]]
        report["recommendation"] = "Do not start training yet. Fix failed priors first: " + ", ".join(failed)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
