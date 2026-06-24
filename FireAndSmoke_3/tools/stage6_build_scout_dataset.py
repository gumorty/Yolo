from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_images(image_dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for image_dir in image_dirs:
        if image_dir.exists():
            images.extend(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def has_label_object(image_path: Path) -> bool:
    label = label_path_for(image_path)
    return label.exists() and bool(label.read_text(encoding="utf-8", errors="ignore").strip())


def copy_pair(image_path: Path, out_dir: Path, split: str, prefix: str, index: int) -> None:
    image_out = out_dir / split / "images"
    label_out = out_dir / split / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    dst_image = image_out / f"{prefix}_{index:06d}{image_path.suffix.lower()}"
    dst_label = label_out / f"{dst_image.stem}.txt"
    shutil.copy2(image_path, dst_image)
    src_label = label_path_for(image_path)
    if src_label.exists():
        shutil.copy2(src_label, dst_label)
    else:
        dst_label.write_text("", encoding="utf-8")


def write_data_yaml(out_dir: Path) -> None:
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "val/images",
        "nc": 3,
        "names": ["fire", "other", "smoke"],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small Stage6 scout dataset for fast training checks.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--base-train", nargs="+", required=True, type=Path)
    parser.add_argument("--base-val", nargs="+", required=True, type=Path)
    parser.add_argument("--hard-train", nargs="*", default=[], type=Path)
    parser.add_argument("--hard-val", nargs="*", default=[], type=Path)
    parser.add_argument("--base-train-count", type=int, default=2500)
    parser.add_argument("--base-val-count", type=int, default=500)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Output exists: {args.out}")
    args.out.mkdir(parents=True)
    rng = random.Random(args.seed)

    base_train = [p for p in read_images(args.base_train) if has_label_object(p)]
    base_val = [p for p in read_images(args.base_val) if has_label_object(p)]
    hard_train = read_images(args.hard_train)
    hard_val = read_images(args.hard_val)
    rng.shuffle(base_train)
    rng.shuffle(base_val)

    copied = {"base_train": 0, "base_val": 0, "hard_train": 0, "hard_val": 0}
    for idx, image in enumerate(base_train[: args.base_train_count]):
        copy_pair(image, args.out, "train", "base", idx)
        copied["base_train"] += 1
    for idx, image in enumerate(hard_train):
        copy_pair(image, args.out, "train", "hard", idx)
        copied["hard_train"] += 1
    for idx, image in enumerate(base_val[: args.base_val_count]):
        copy_pair(image, args.out, "val", "base", idx)
        copied["base_val"] += 1
    for idx, image in enumerate(hard_val):
        copy_pair(image, args.out, "val", "hard", idx)
        copied["hard_val"] += 1

    write_data_yaml(args.out)
    (args.out / "build_stats.yaml").write_text(yaml.safe_dump(copied, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(copied, sort_keys=False))
    print(args.out / "data.yaml")


if __name__ == "__main__":
    main()
