from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {
    "train": ["train", "Train", "training", "Training"],
    "val": ["val", "valid", "Val", "Valid", "validation", "Validation"],
    "test": ["test", "Test", "testing", "Testing"],
}


def label_path_for(image_path: Path, image_root: Path, label_root: Path) -> Path:
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def find_split_dir(root: Path, split: str, kind: str) -> Path | None:
    for alias in SPLIT_ALIASES[split]:
        candidates = [
            root / alias / kind,
            root / kind / alias,
            root / alias,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def infer_dirs(root: Path, split: str) -> tuple[Path | None, Path | None]:
    image_dir = find_split_dir(root, split, "images")
    label_dir = find_split_dir(root, split, "labels")
    if not label_dir:
        label_dir = find_split_dir(root, split, "annotations")
    return image_dir, label_dir


def parse_class_map(items: list[str]) -> dict[int, int | None]:
    mapping: dict[int, int | None] = {}
    for item in items:
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        src = int(left)
        if right.lower() in {"drop", "ignore", "none"}:
            mapping[src] = None
        else:
            mapping[src] = int(right)
    return mapping


def convert_label(label_path: Path, out_label: Path, class_map: dict[int, int | None]) -> int:
    rows: list[str] = []
    if label_path.exists():
        for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                src_cls = int(float(parts[0]))
            except ValueError:
                continue
            dst_cls = class_map.get(src_cls, src_cls)
            if dst_cls is None:
                continue
            rows.append(" ".join([str(dst_cls), *parts[1:5]]))
    out_label.parent.mkdir(parents=True, exist_ok=True)
    out_label.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def copy_split(
    split: str,
    image_dir: Path,
    label_dir: Path,
    out: Path,
    source_id: str,
    class_map: dict[int, int | None],
    include_empty: bool,
) -> dict[str, int]:
    out_images = out / split / "images"
    out_labels = out / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    stats = {"images_seen": 0, "images_copied": 0, "empty_labels": 0, "objects": 0, "missing_labels": 0}
    for image_path in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS):
        stats["images_seen"] += 1
        label_path = label_path_for(image_path, image_dir, label_dir)
        if not label_path.exists():
            stats["missing_labels"] += 1
            if not include_empty:
                continue
        stem = f"{source_id}_{split}_{image_path.stem}"
        out_image = out_images / f"{stem}{image_path.suffix.lower()}"
        out_label = out_labels / f"{stem}.txt"
        objects = convert_label(label_path, out_label, class_map)
        if objects == 0:
            stats["empty_labels"] += 1
            if not include_empty:
                out_label.unlink(missing_ok=True)
                continue
        shutil.copy2(image_path, out_image)
        stats["images_copied"] += 1
        stats["objects"] += objects
    return stats


def copy_split_from_list(
    split: str,
    list_path: Path,
    root: Path,
    image_dir: Path,
    label_dir: Path,
    out: Path,
    source_id: str,
    class_map: dict[int, int | None],
    include_empty: bool,
) -> dict[str, int]:
    out_images = out / split / "images"
    out_labels = out / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    stats = {"images_seen": 0, "images_copied": 0, "empty_labels": 0, "objects": 0, "missing_labels": 0}
    for raw in list_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        item = raw.strip()
        if not item:
            continue
        image_path = Path(item)
        if not image_path.is_absolute():
            image_path = (root / item.replace("./", "")).resolve()
        if not image_path.exists():
            alt = image_dir / Path(item).name
            image_path = alt.resolve()
        stats["images_seen"] += 1
        if not image_path.exists():
            stats["missing_labels"] += 1
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"] += 1
            if not include_empty:
                continue
        stem = f"{source_id}_{split}_{image_path.stem}"
        out_image = out_images / f"{stem}{image_path.suffix.lower()}"
        out_label = out_labels / f"{stem}.txt"
        objects = convert_label(label_path, out_label, class_map)
        if objects == 0:
            stats["empty_labels"] += 1
            if not include_empty:
                out_label.unlink(missing_ok=True)
                continue
        shutil.copy2(image_path, out_image)
        stats["images_copied"] += 1
        stats["objects"] += objects
    return stats


def write_data_yaml(out: Path, names: list[str]) -> None:
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    data = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize a YOLO-style source dataset into Stage4 source layout.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--names", nargs="+", default=["fire", "smoke"])
    parser.add_argument("--class-map", nargs="*", default=[], help="Example: 0=0 1=2 to map fire/smoke into fire/other/smoke ids.")
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--image-dir", type=Path, help="Flat image directory, for datasets that use train.txt/val.txt/test.txt lists.")
    parser.add_argument("--label-dir", type=Path, help="Flat label directory matching --image-dir by stem.")
    parser.add_argument("--split-list-dir", type=Path, help="Directory containing train.txt/val.txt/test.txt split lists.")
    parser.add_argument("--test-as-val-if-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    out = args.out.resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {out}. Pass --overwrite to rebuild.")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    class_map = parse_class_map(args.class_map)
    stats = {}
    for split in ("train", "val", "test"):
        if args.split_list_dir:
            split_list_dir = args.split_list_dir if args.split_list_dir.is_absolute() else (root / args.split_list_dir)
            list_path = split_list_dir / f"{split}.txt"
            image_dir = args.image_dir if args.image_dir and args.image_dir.is_absolute() else (root / args.image_dir) if args.image_dir else root / "images"
            label_dir = args.label_dir if args.label_dir and args.label_dir.is_absolute() else (root / args.label_dir) if args.label_dir else root / "labels"
            if not list_path.exists():
                stats[split] = {"skipped": True, "reason": f"missing split list: {list_path}"}
                continue
            stats[split] = copy_split_from_list(split, list_path, root, image_dir, label_dir, out, args.source_id, class_map, args.include_empty)
        else:
            image_dir, label_dir = infer_dirs(root, split)
            if not image_dir or not label_dir:
                stats[split] = {"skipped": True, "reason": "missing image or label directory"}
                continue
            stats[split] = copy_split(split, image_dir, label_dir, out, args.source_id, class_map, args.include_empty)
    if args.test_as_val_if_missing and not any((out / "val" / "images").glob("*")) and any((out / "test" / "images").glob("*")):
        stats["val_from_test"] = copy_split("val", out / "test" / "images", out / "test" / "labels", out, args.source_id, {}, True)
    write_data_yaml(out, args.names)
    (out / "prepare_stats.yaml").write_text(yaml.safe_dump(stats, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(stats, allow_unicode=True, sort_keys=False))
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
