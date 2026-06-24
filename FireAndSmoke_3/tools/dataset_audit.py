from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _resolve_many(base: Path, value) -> list[Path]:
    if isinstance(value, list):
        return [_resolve(base, str(item)) for item in value]
    return [_resolve(base, str(value))]


def _label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        rows.append((cls, x, y, w, h))
    return rows


def _area_bin(w: float, h: float) -> str:
    area = w * h
    if area < 0.001:
        return "tiny(<0.1%)"
    if area < 0.01:
        return "small(0.1-1%)"
    if area < 0.05:
        return "medium(1-5%)"
    return "large(>=5%)"


def _image_size(image_path: Path) -> tuple[int, int] | None:
    if Image is None:
        return None
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return None


def audit_dataset(data_yaml: Path) -> dict:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    yaml_dir = data_yaml.parent
    names = data.get("names", [])
    if isinstance(names, dict):
        names_map = {int(k): str(v) for k, v in names.items()}
    else:
        names_map = {i: str(v) for i, v in enumerate(names)}

    out = {
        "data_yaml": str(data_yaml),
        "names": names_map,
        "splits": {},
        "totals": {
            "images": 0,
            "missing_labels": 0,
            "empty_labels": 0,
            "instances": {},
            "area_bins": {},
            "class_area_bins": {},
            "image_shapes": {},
        },
    }

    total_instances = Counter()
    total_bins = Counter()
    total_class_bins = Counter()
    total_shapes = Counter()
    for split in ("train", "val", "valid", "test"):
        if split not in data:
            continue
        image_dirs = _resolve_many(yaml_dir, data[split])
        images = []
        for image_dir in image_dirs:
            if image_dir.exists():
                images.extend(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        images = sorted(images)
        instances = Counter()
        area_bins = Counter()
        class_area_bins = Counter()
        image_shapes = Counter()
        missing = 0
        empty = 0
        duplicate_labels = 0
        invalid_labels = 0

        for image_path in images:
            shape = _image_size(image_path)
            if shape is not None:
                w, h = shape
                if max(w, h) <= 640:
                    image_shapes["max_side<=640"] += 1
                elif max(w, h) <= 960:
                    image_shapes["641-960"] += 1
                elif max(w, h) <= 1280:
                    image_shapes["961-1280"] += 1
                else:
                    image_shapes[">1280"] += 1
            label_path = _label_path_for(image_path)
            if not label_path.exists():
                missing += 1
                continue
            labels = _read_labels(label_path)
            if not labels:
                empty += 1
            seen = set()
            for cls, x, y, w, h in labels:
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    invalid_labels += 1
                    continue
                key = (cls, round(x, 6), round(y, 6), round(w, 6), round(h, 6))
                if key in seen:
                    duplicate_labels += 1
                seen.add(key)
                class_name = names_map.get(cls, str(cls))
                bin_name = _area_bin(w, h)
                instances[class_name] += 1
                area_bins[bin_name] += 1
                class_area_bins[f"{class_name}|{bin_name}"] += 1

        split_key = "valid" if split == "val" else split
        out["splits"][split_key] = {
            "image_dir": [str(path) for path in image_dirs],
            "images": len(images),
            "missing_labels": missing,
            "empty_labels": empty,
            "duplicate_labels": duplicate_labels,
            "invalid_labels": invalid_labels,
            "instances": dict(instances),
            "area_bins": dict(area_bins),
            "class_area_bins": dict(class_area_bins),
            "image_shapes": dict(image_shapes),
        }
        out["totals"]["images"] += len(images)
        out["totals"]["missing_labels"] += missing
        out["totals"]["empty_labels"] += empty
        total_instances.update(instances)
        total_bins.update(area_bins)
        total_class_bins.update(class_area_bins)
        total_shapes.update(image_shapes)

    out["totals"]["instances"] = dict(total_instances)
    out["totals"]["area_bins"] = dict(total_bins)
    out["totals"]["class_area_bins"] = dict(total_class_bins)
    out["totals"]["image_shapes"] = dict(total_shapes)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit_dataset(args.data)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
