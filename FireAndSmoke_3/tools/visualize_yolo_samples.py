from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COLORS = {
    0: (30, 60, 230),
    1: (60, 180, 60),
    2: (230, 160, 30),
}


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {idx: str(value) for idx, value in enumerate(names or [])}


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def draw_labels(image_path: Path, names: dict[int, str], size: int) -> Any | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    h, w = image.shape[:2]
    label_path = label_path_for(image_path)
    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, bw, bh = (float(v) for v in parts[1:5])
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            color = COLORS.get(cls, (200, 200, 200))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, w // 500))
            cv2.putText(image, names.get(cls, str(cls)), (max(0, x1), max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    scale = size / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a contact sheet of YOLO labels for manual sanity checks.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--tile-size", type=int, default=360)
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    image_dir = resolve(base, str(data.get(args.split, data.get("val"))))
    names = names_map(data.get("names", []))
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(args.seed)
    rng.shuffle(images)
    thumbs = []
    for image_path in images:
        label_path = label_path_for(image_path)
        if not label_path.exists() or not label_path.read_text(encoding="utf-8", errors="ignore").strip():
            continue
        thumb = draw_labels(image_path, names, args.tile_size)
        if thumb is not None:
            thumbs.append(thumb)
        if len(thumbs) >= args.samples:
            break
    if not thumbs:
        raise SystemExit("No labeled samples found.")
    cols = int(args.samples**0.5)
    cols = max(1, cols)
    rows = (len(thumbs) + cols - 1) // cols
    canvas = 255 * cv2.UMat(args.tile_size * rows, args.tile_size * cols, cv2.CV_8UC3).get()
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        h, w = thumb.shape[:2]
        canvas[row * args.tile_size : row * args.tile_size + h, col * args.tile_size : col * args.tile_size + w] = thumb
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), canvas)
    print(args.out)


if __name__ == "__main__":
    main()
