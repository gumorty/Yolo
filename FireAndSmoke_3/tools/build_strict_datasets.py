from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


def write_yaml(path: Path, train: Path, val: Path, test: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "train": str(train),
        "val": str(val),
        "test": str(test),
        "nc": len(names),
        "names": names,
    }
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def require_split(dataset: Path, split: str) -> Path:
    aliases = {"val": ["valid", "val"], "train": ["train"], "test": ["test"]}
    for name in aliases[split]:
        image_dir = dataset / name / "images"
        if image_dir.exists():
            return image_dir.resolve()
        alt_image_dir = dataset / "images" / name
        if alt_image_dir.exists():
            return alt_image_dir.resolve()
    raise FileNotFoundError(f"Missing {split}/images under {dataset}")


def remove_stale_yaml_dir(path: Path, source: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed stale generated dataset config: {path} (missing source: {source})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict non-blended dataset YAMLs for round 3.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--sensors3", type=Path)
    parser.add_argument("--sensors2", type=Path)
    parser.add_argument("--self2", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    datasets = root / "datasets"
    sensors3 = args.sensors3 or datasets / "raw_yolov8" / "yolov8"
    sensors2 = args.sensors2 or datasets / "binary_fire_smoke"

    created = []
    stale_targets: list[tuple[Path, Path]] = []
    if sensors3.exists():
        out_dir = datasets / "strict_sensors_3cls"
        out = out_dir / "data.yaml"
        write_yaml(
            out,
            require_split(sensors3, "train"),
            require_split(sensors3, "val"),
            require_split(sensors3, "test"),
            ["fire", "other", "smoke"],
        )
        created.append(out)
    else:
        stale_targets.append((datasets / "strict_sensors_3cls", sensors3))

    if sensors2.exists():
        out_dir = datasets / "strict_sensors_2cls"
        out = out_dir / "data.yaml"
        write_yaml(
            out,
            require_split(sensors2, "train"),
            require_split(sensors2, "val"),
            require_split(sensors2, "test"),
            ["fire", "smoke"],
        )
        created.append(out)
    else:
        stale_targets.append((datasets / "strict_sensors_2cls", sensors2))

    if args.self2 and args.self2.exists():
        out_dir = datasets / "self_fire_pic_2cls_standalone"
        out = out_dir / "data.yaml"
        write_yaml(
            out,
            require_split(args.self2, "train"),
            require_split(args.self2, "val"),
            require_split(args.self2, "test"),
            ["fire", "smoke"],
        )
        created.append(out)
    elif args.self2:
        stale_targets.append((datasets / "self_fire_pic_2cls_standalone", args.self2))

    for target, source in stale_targets:
        remove_stale_yaml_dir(target, source)

    if not created:
        raise SystemExit("No dataset YAMLs created. Check dataset paths.")

    print("Created strict dataset YAMLs:")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
