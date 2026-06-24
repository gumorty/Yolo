from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def is_empty_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    return not label_path.read_text(encoding="utf-8", errors="ignore").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed empty-label false-alarm evaluation set.")
    parser.add_argument("--source-images", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--prefix", default="false_alarm_v2")
    args = parser.parse_args()

    if not args.source_images.exists():
        raise SystemExit(f"source image directory not found: {args.source_images}")

    candidates = []
    for image_path in sorted(args.source_images.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = label_for_image(image_path)
        if is_empty_label(label_path):
            candidates.append((image_path, label_path))

    if not candidates:
        raise SystemExit(f"no empty-label candidates found under: {args.source_images}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: min(args.max_images, len(candidates))]

    image_out = args.out_dir / "images"
    label_out = args.out_dir / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, (image_path, label_path) in enumerate(selected, start=1):
        dst_name = f"{args.prefix}_{idx:05d}{image_path.suffix.lower()}"
        dst_image = image_out / dst_name
        dst_label = label_out / Path(dst_name).with_suffix(".txt").name
        shutil.copy2(image_path, dst_image)
        dst_label.write_text("", encoding="utf-8")
        manifest.append(
            {
                "image": str(dst_image),
                "label": str(dst_label),
                "source_image": str(image_path),
                "source_label": str(label_path),
            }
        )

    data_yaml = "\n".join(
        [
            "path: .",
            "train: images",
            "val: images",
            "test: images",
            "nc: 3",
            "names:",
            "- fire",
            "- other",
            "- smoke",
            "",
        ]
    )
    (args.out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# UAV 误报评估集",
                "",
                f"该固定评估集包含 {len(selected)} 张空标签图像。",
                "",
                "数据来源：",
                f"- `{args.source_images}`",
                "",
                "用途：",
                "",
                "- 评估模型在已验证空标签无人机火焰/烟雾背景图像上的误报情况。",
                "- 该数据集只用于测试，不进入训练。",
                "- 需要报告 FP/image、FP/100 images、FP/1000 images 以及误报类别分布。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary = {
        "source_images": str(args.source_images),
        "out_dir": str(args.out_dir),
        "candidate_empty_images": len(candidates),
        "selected": len(selected),
        "seed": args.seed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
