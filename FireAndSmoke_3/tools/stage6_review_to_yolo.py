from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


DATA_YAML = """train: train/images
val: val/images
test: val/images
nc: 3
names:
- fire
- other
- smoke
"""


def normalize_decision(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def copy_empty(row: dict[str, str], split: str, out_dir: Path, index: int) -> dict[str, str]:
    src = Path(row["review_image"])
    image_dir = out_dir / split / "images"
    label_dir = out_dir / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() or ".jpg"
    dst = image_dir / f"stage6_hardneg_{index:06d}{suffix}"
    shutil.copy2(src, dst)
    (label_dir / f"{dst.stem}.txt").write_text("", encoding="utf-8")
    return {"source": str(src), "image": str(dst), "label": str(label_dir / f"{dst.stem}.txt"), "split": split}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert reviewed Stage6 hard-negative rows into a YOLO empty-label dataset.")
    parser.add_argument("--manifest", required=True, type=Path, help="Edited review_manifest.csv from stage6_mine_hard_cases.py.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--decision", default="confirmed_empty", help="Review decision value to export as empty hard negative.")
    args = parser.parse_args()

    rows = list(csv.DictReader(args.manifest.open("r", encoding="utf-8-sig")))
    selected = [row for row in rows if normalize_decision(row.get("review_decision", "")) == normalize_decision(args.decision)]
    if args.out.exists():
        raise SystemExit(f"Output exists: {args.out}. Remove it or choose a new output directory.")
    args.out.mkdir(parents=True)

    copied = []
    val_cut = max(1, int(len(selected) * args.val_ratio)) if selected else 0
    for idx, row in enumerate(selected):
        split = "val" if idx < val_cut else "train"
        copied.append(copy_empty(row, split, args.out, idx))

    (args.out / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    summary = {
        "manifest": str(args.manifest),
        "selected_decision": args.decision,
        "selected_rows": len(selected),
        "train_images": sum(1 for item in copied if item["split"] == "train"),
        "val_images": sum(1 for item in copied if item["split"] == "val"),
        "data_yaml": str(args.out / "data.yaml"),
    }
    (args.out / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
