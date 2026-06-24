from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"val": ["val", "valid"], "valid": ["valid", "val"], "train": ["train"], "test": ["test"]}


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


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


def split_image_dir(data: dict[str, Any], base: Path, split: str) -> Path | None:
    for key in SPLIT_ALIASES.get(split, [split]):
        if key in data:
            return resolve(base, str(data[key]))
    return None


def read_source(data_yaml: Path) -> tuple[dict[str, Path], dict[int, str]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = data_yaml.parent
    split_dirs = {}
    for split in ("train", "val", "test"):
        image_dir = split_image_dir(data, base, split)
        if image_dir:
            split_dirs[split] = image_dir
    return split_dirs, names_map(data.get("names", []))


def convert_label_rows(
    label_path: Path,
    source_names: dict[int, str],
    class_map: dict[str, str],
    target_index: dict[str, int],
) -> list[str]:
    if not label_path.exists():
        return []
    rows: list[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            source_cls = int(float(parts[0]))
        except ValueError:
            continue
        source_name = source_names.get(source_cls, str(source_cls))
        target_name = class_map.get(source_name, class_map.get(str(source_cls), ""))
        if not target_name or target_name.lower() in {"drop", "ignore", "none"}:
            continue
        if target_name not in target_index:
            raise ValueError(f"Class map target '{target_name}' is not in target_names")
        rows.append(" ".join([str(target_index[target_name]), *parts[1:5]]))
    return rows


def copy_split(
    source_id: str,
    source_split: str,
    target_split: str,
    image_dir: Path,
    out_dir: Path,
    source_names: dict[int, str],
    class_map: dict[str, str],
    target_index: dict[str, int],
    include_empty: bool,
    max_images: int,
    rng: random.Random,
) -> dict[str, int]:
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_images and len(images) > max_images:
        images = rng.sample(images, max_images)
        images.sort()
    out_images = out_dir / target_split / "images"
    out_labels = out_dir / target_split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    stats = {"seen": len(images), "copied": 0, "empty": 0, "dropped_empty": 0}
    for image_path in images:
        rows = convert_label_rows(label_path_for(image_path), source_names, class_map, target_index)
        if not rows:
            if include_empty:
                stats["empty"] += 1
            else:
                stats["dropped_empty"] += 1
                continue
        safe_stem = f"{source_id}_{source_split}_{image_path.stem}"
        out_image = out_images / f"{safe_stem}{image_path.suffix.lower()}"
        out_label = out_labels / f"{safe_stem}.txt"
        shutil.copy2(image_path, out_image)
        out_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        stats["copied"] += 1
    return stats


def write_data_yaml(out_dir: Path, target_names: list[str]) -> None:
    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(target_names),
        "names": target_names,
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge curated YOLO datasets into one Stage4 fire/other/smoke dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_names = [str(v) for v in config["target_names"]]
    target_index = {name: idx for idx, name in enumerate(target_names)}
    out_dir = args.out.resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out_dir}. Pass --overwrite to rebuild.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rng = random.Random(int(config.get("seed", 20260609)))
    stats: dict[str, Any] = {}
    for source in config.get("sources", []):
        if not source.get("enabled", False):
            continue
        source_id = str(source["id"])
        data_yaml = resolve(config_path.parent, str(source["data"]))
        if not data_yaml.exists():
            raise FileNotFoundError(f"{source_id}: missing data yaml: {data_yaml}")
        split_dirs, source_names = read_source(data_yaml)
        class_map = {str(k): str(v) for k, v in source.get("class_map", {}).items()}
        include_empty = bool(source.get("include_empty", False))
        max_images = source.get("max_images", {}) or {}
        stats[source_id] = {}
        for split in source.get("splits", ["train", "val"]):
            source_split = "val" if split == "valid" else str(split)
            if source_split not in split_dirs:
                continue
            limit = int(max_images.get(source_split, max_images.get(split, 0)) or 0)
            stats[source_id][source_split] = copy_split(
                source_id=source_id,
                source_split=source_split,
                target_split=source_split,
                image_dir=split_dirs[source_split],
                out_dir=out_dir,
                source_names=source_names,
                class_map=class_map,
                target_index=target_index,
                include_empty=include_empty,
                max_images=limit,
                rng=rng,
            )
    write_data_yaml(out_dir, target_names)
    (out_dir / "mix_stats.yaml").write_text(yaml.safe_dump(stats, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(stats, allow_unicode=True, sort_keys=False))
    print(f"Wrote {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
