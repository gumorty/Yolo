from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_shape(path: Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired RGB/Thermal images before RGB-T training.")
    parser.add_argument("--rgb-dir", required=True, type=Path)
    parser.add_argument("--thermal-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rgb = {p.stem: p for p in args.rgb_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    thermal = {p.stem: p for p in args.thermal_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    common = sorted(set(rgb) & set(thermal))
    report = {
        "rgb_dir": str(args.rgb_dir),
        "thermal_dir": str(args.thermal_dir),
        "rgb_images": len(rgb),
        "thermal_images": len(thermal),
        "paired": len(common),
        "missing_thermal": sorted(set(rgb) - set(thermal))[:100],
        "missing_rgb": sorted(set(thermal) - set(rgb))[:100],
        "shape_mismatch": [],
    }
    for stem in common[:5000]:
        r_shape = image_shape(rgb[stem])
        t_shape = image_shape(thermal[stem])
        if r_shape and t_shape and r_shape != t_shape:
            report["shape_mismatch"].append({"stem": stem, "rgb": r_shape, "thermal": t_shape})
            if len(report["shape_mismatch"]) >= 100:
                break

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
