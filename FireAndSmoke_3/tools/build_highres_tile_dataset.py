from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Label:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(label_path: Path, width: int, height: int) -> list[Label]:
    if not label_path.exists():
        return []
    labels: list[Label] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        x1 = max(0.0, (x - w / 2.0) * width)
        y1 = max(0.0, (y - h / 2.0) * height)
        x2 = min(float(width), (x + w / 2.0) * width)
        y2 = min(float(height), (y + h / 2.0) * height)
        if x2 > x1 and y2 > y1:
            labels.append(Label(cls, x1, y1, x2, y2))
    return labels


def format_tile_label(label: Label, tile: tuple[int, int, int, int]) -> str | None:
    tx1, ty1, tx2, ty2 = tile
    ix1 = max(label.x1, tx1)
    iy1 = max(label.y1, ty1)
    ix2 = min(label.x2, tx2)
    iy2 = min(label.y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    tw = tx2 - tx1
    th = ty2 - ty1
    cx = ((ix1 + ix2) / 2.0 - tx1) / tw
    cy = ((iy1 + iy2) / 2.0 - ty1) / th
    bw = (ix2 - ix1) / tw
    bh = (iy2 - iy1) / th
    return f"{label.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def visibility(label: Label, tile: tuple[int, int, int, int]) -> float:
    tx1, ty1, tx2, ty2 = tile
    ix1 = max(label.x1, tx1)
    iy1 = max(label.y1, ty1)
    ix2 = min(label.x2, tx2)
    iy2 = min(label.y2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1.0, (label.x2 - label.x1) * (label.y2 - label.y1))
    return inter / area


def iter_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    if width <= tile_size and height <= tile_size:
        return [(0, 0, width, height)]
    stride = max(1, int(tile_size * (1.0 - overlap)))
    max_x = max(width - tile_size, 0)
    max_y = max(height - tile_size, 0)
    xs = list(range(0, max_x + 1, stride))
    ys = list(range(0, max_y + 1, stride))
    if xs[-1] != max_x:
        xs.append(max_x)
    if ys[-1] != max_y:
        ys.append(max_y)
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height)) for y in ys for x in xs]


def write_yaml(path: Path, split_dirs: dict[str, Path], names: list[str]) -> None:
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def split_image_dirs(data_yaml: Path) -> tuple[dict[str, Path], list[str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    names_raw = data.get("names", [])
    if isinstance(names_raw, dict):
        names = [str(names_raw[i]) for i in sorted(names_raw)]
    else:
        names = [str(v) for v in names_raw]
    dirs = {
        "train": resolve(base, str(data["train"])),
        "val": resolve(base, str(data.get("val", data.get("valid")))),
        "test": resolve(base, str(data.get("test", data.get("val", data.get("valid"))))),
    }
    return dirs, names


def should_keep_tile(
    labels: list[Label],
    tile: tuple[int, int, int, int],
    target_classes: set[int],
    min_visibility: float,
) -> tuple[bool, list[str]]:
    rows: list[str] = []
    has_target = False
    for label in labels:
        if visibility(label, tile) < min_visibility:
            continue
        row = format_tile_label(label, tile)
        if row is None:
            continue
        rows.append(row)
        if not target_classes or label.cls in target_classes:
            has_target = True
    return has_target, rows


def copy_full_image(image_path: Path, labels: list[Label], out_images: Path, out_labels: Path, width: int, height: int) -> None:
    out_image = out_images / f"full_{image_path.stem}{image_path.suffix.lower()}"
    out_label = out_labels / f"full_{image_path.stem}.txt"
    shutil.copy2(image_path, out_image)
    rows = []
    for label in labels:
        cx = ((label.x1 + label.x2) / 2.0) / width
        cy = ((label.y1 + label.y2) / 2.0) / height
        bw = (label.x2 - label.x1) / width
        bh = (label.y2 - label.y1) / height
        rows.append(f"{label.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    out_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def build_split(
    split: str,
    image_dir: Path,
    out_split: Path,
    names: list[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, int]:
    out_images = out_split / "images"
    out_labels = out_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    target_classes = {names.index(name) for name in args.positive_classes if name in names}
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    stats = {"source_images": len(images), "full_images": 0, "positive_tiles": 0, "negative_tiles": 0, "skipped_unreadable": 0}

    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped_unreadable"] += 1
            continue
        height, width = image.shape[:2]
        labels = read_yolo_labels(label_path_for(image_path), width, height)
        if args.copy_full:
            copy_full_image(image_path, labels, out_images, out_labels, width, height)
            stats["full_images"] += 1

        for tile_idx, tile in enumerate(iter_tiles(width, height, args.tile_size, args.overlap)):
            keep_positive, rows = should_keep_tile(labels, tile, target_classes, args.min_visibility)
            keep_negative = not keep_positive and args.negative_ratio > 0 and rng.random() < args.negative_ratio
            if not keep_positive and not keep_negative:
                continue
            x1, y1, x2, y2 = tile
            crop = image[y1:y2, x1:x2]
            stem = f"tile_{image_path.stem}_{tile_idx:04d}_{x1}_{y1}_{x2}_{y2}"
            cv2.imwrite(str(out_images / f"{stem}.jpg"), crop)
            (out_labels / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            if keep_positive:
                stats["positive_tiles"] += 1
            else:
                stats["negative_tiles"] += 1
    print(f"{split}: {stats}")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO full+tile dataset for high-resolution small fire/smoke training.")
    parser.add_argument("--data", required=True, type=Path, help="Source YOLO data.yaml.")
    parser.add_argument("--out", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--positive-classes", nargs="*", default=["fire", "smoke"])
    parser.add_argument("--negative-ratio", type=float, default=0.10, help="Probability of keeping empty/background tiles.")
    parser.add_argument("--copy-full", action="store_true", help="Also include original full images in each split.")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out}. Pass --overwrite to rebuild.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    split_dirs, names = split_image_dirs(data_yaml)
    rng = random.Random(args.seed)
    stats = {}
    for split, image_dir in split_dirs.items():
        stats[split] = build_split(split, image_dir, out / split, names, args, rng)
    write_yaml(out / "data.yaml", {split: out / split for split in split_dirs}, names)
    (out / "build_stats.yaml").write_text(yaml.safe_dump(stats, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
