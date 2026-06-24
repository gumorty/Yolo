from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": ["train"], "val": ["val", "valid"], "valid": ["valid", "val"], "test": ["test"]}


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(value) for idx, value in enumerate(names or [])}


def split_image_dir(data: dict[str, Any], base: Path, split: str) -> Path | None:
    for key in SPLIT_ALIASES.get(split, [split]):
        if key in data:
            return resolve(base, str(data[key]))
    return None


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def area_bin(width: float, height: float) -> str:
    area = width * height
    if area < 0.001:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def read_labels(label_path: Path, source_names: dict[int, str]) -> list[dict[str, str]]:
    if not label_path.exists():
        return []
    labels: list[dict[str, str]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        labels.append({"class_name": source_names.get(cls, str(cls)), "size": area_bin(w, h)})
    return labels


def copy_item(image_path: Path, out_images: Path, out_labels: Path, prefix: str) -> None:
    stem = f"{prefix}_{image_path.stem}"
    out_image = out_images / f"{stem}{image_path.suffix.lower()}"
    out_label = out_labels / f"{stem}.txt"
    shutil.copy2(image_path, out_image)
    label_path = label_path_for(image_path)
    if label_path.exists():
        shutil.copy2(label_path, out_label)
    else:
        out_label.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed YOLO holdout set focused on tiny/small fire/smoke and hard negatives.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    parser.add_argument("--target-classes", nargs="+", default=["fire", "smoke"])
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--min-small-positive-ratio", type=float, default=0.70)
    parser.add_argument("--negative-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    source_names = names_map(data.get("names", []))
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out}. Pass --overwrite to rebuild.")
        shutil.rmtree(out)
    out_images = out / "images"
    out_labels = out / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    target_classes = set(args.target_classes)
    positives: list[Path] = []
    negatives: list[Path] = []
    regular_positives: list[Path] = []
    for split in args.splits:
        image_dir = split_image_dir(data, base, split)
        if not image_dir or not image_dir.exists():
            continue
        for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
            labels = read_labels(label_path_for(image_path), source_names)
            has_target = any(item["class_name"] in target_classes for item in labels)
            has_small_target = any(item["class_name"] in target_classes and item["size"] in {"tiny", "small"} for item in labels)
            if has_small_target:
                positives.append(image_path)
            elif has_target:
                regular_positives.append(image_path)
            else:
                negatives.append(image_path)

    rng = random.Random(args.seed)
    rng.shuffle(positives)
    rng.shuffle(regular_positives)
    rng.shuffle(negatives)

    desired_small = int(args.max_images * args.min_small_positive_ratio)
    desired_negative = int(args.max_images * args.negative_ratio)
    selected = positives[:desired_small]
    remaining_slots = max(args.max_images - len(selected), 0)
    selected.extend(regular_positives[: max(remaining_slots - desired_negative, 0)])
    remaining_slots = max(args.max_images - len(selected), 0)
    selected.extend(negatives[:remaining_slots])
    selected = selected[: args.max_images]

    manifest = {
        "data": str(data_yaml),
        "target_classes": sorted(target_classes),
        "max_images": args.max_images,
        "candidate_small_positive": len(positives),
        "candidate_regular_positive": len(regular_positives),
        "candidate_negative": len(negatives),
        "selected": [],
    }
    for image_path in selected:
        copy_item(image_path, out_images, out_labels, "holdout")
        labels = read_labels(label_path_for(image_path), source_names)
        manifest["selected"].append({"image": str(image_path), "labels": labels})

    data_out = {
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": len(source_names),
        "names": [source_names[idx] for idx in sorted(source_names)],
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected"} | {"selected_count": len(selected)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
