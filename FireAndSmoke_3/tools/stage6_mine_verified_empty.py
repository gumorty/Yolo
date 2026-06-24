from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "fire", 1: "other", 2: "smoke"}


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Cannot import ultralytics.YOLO: {exc}") from exc
    return YOLO


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


def is_empty_label(image_path: Path) -> bool:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        return False
    return not label_path.read_text(encoding="utf-8", errors="ignore").strip()


def read_split_images(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    value = data.get(split)
    if value is None and split == "val":
        value = data.get("valid")
    if value is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")
    image_dir = resolve(data_yaml.parent, str(value))
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def predict(model: Any, image_path: Path, imgsz: int, conf: float, iou: float) -> list[dict[str, Any]]:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    clss = [int(v) for v in result.boxes.cls.cpu().tolist()]
    confs = [float(v) for v in result.boxes.conf.cpu().tolist()]
    return [
        {
            "class_id": cls,
            "class_name": CLASS_NAMES.get(cls, str(cls)),
            "confidence": round(score, 5),
            "xyxy": [round(float(v), 2) for v in box],
        }
        for box, cls, score in zip(boxes, clss, confs)
    ]


def draw(image_path: Path, preds: list[dict[str, Any]], out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    colors = {0: (0, 80, 255), 1: (255, 180, 0), 2: (180, 180, 180)}
    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred["xyxy"]]
        color = colors.get(int(pred["class_id"]), (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{pred["class_name"]} {pred["confidence"]:.2f}'
        cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def copy_to_dataset(image_path: Path, out_dir: Path, split: str, index: int) -> tuple[Path, Path]:
    image_dir = out_dir / split / "images"
    label_dir = out_dir / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    dst_image = image_dir / f"verified_empty_fp_{index:06d}{image_path.suffix.lower()}"
    dst_label = label_dir / f"{dst_image.stem}.txt"
    shutil.copy2(image_path, dst_image)
    dst_label.write_text("", encoding="utf-8")
    return dst_image, dst_label


def write_data_yaml(out_dir: Path) -> None:
    (out_dir / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: val/images\nnc: 3\nnames:\n- fire\n- other\n- smoke\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine verified-empty hard negatives from existing YOLO empty-label images.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--max-empty", type=int, default=5000)
    parser.add_argument("--max-export", type=int, default=2000)
    parser.add_argument("--min-models", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Output exists: {args.out}. Choose a new path or remove it first.")
    args.out.mkdir(parents=True)

    model_names = args.model_names or [p.stem for p in args.models]
    if len(model_names) != len(args.models):
        raise SystemExit("--model-names length must match --models length")

    YOLO = import_yolo()
    models = [(name, YOLO(str(path))) for name, path in zip(model_names, args.models)]
    images = [p for p in read_split_images(args.data, args.split) if is_empty_label(p)]
    rng = random.Random(args.seed)
    rng.shuffle(images)
    if args.max_empty:
        images = images[: args.max_empty]

    rows: list[dict[str, Any]] = []
    export_index = 0
    for scan_index, image_path in enumerate(images, start=1):
        per_model: dict[str, list[dict[str, Any]]] = {}
        for model_name, model in models:
            preds = predict(model, image_path, args.imgsz, args.conf, args.iou)
            if preds:
                per_model[model_name] = preds
        if len(per_model) < args.min_models:
            if args.progress_every and scan_index % args.progress_every == 0:
                print(f"scanned={scan_index} exported={export_index}", flush=True)
            continue

        val_stride = max(2, round(1.0 / args.val_ratio)) if args.val_ratio > 0 else 0
        split = "val" if val_stride and (export_index + 1) % val_stride == 0 else "train"
        dst_image, dst_label = copy_to_dataset(image_path, args.out, split, export_index)
        annotated_path = ""
        if args.save_annotated:
            first_preds = next(iter(per_model.values()))
            annotated = args.out / "audit_annotated" / f"{dst_image.stem}.jpg"
            draw(image_path, first_preds, annotated)
            annotated_path = str(annotated)

        rows.append(
            {
                "source_image": str(image_path),
                "export_image": str(dst_image),
                "export_label": str(dst_label),
                "split": split,
                "triggered_models": ";".join(per_model.keys()),
                "model_count": len(per_model),
                "prediction_count": sum(len(v) for v in per_model.values()),
                "max_confidence": max(pred["confidence"] for preds in per_model.values() for pred in preds),
                "predictions_json": json.dumps(per_model, ensure_ascii=False),
                "annotated_image": annotated_path,
            }
        )
        export_index += 1
        if args.progress_every and (scan_index % args.progress_every == 0 or export_index % args.progress_every == 0):
            print(f"scanned={scan_index} exported={export_index}", flush=True)
        if args.max_export and export_index >= args.max_export:
            break

    write_data_yaml(args.out)
    with (args.out / "verified_empty_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "source_image",
            "export_image",
            "export_label",
            "split",
            "triggered_models",
            "model_count",
            "prediction_count",
            "max_confidence",
            "annotated_image",
            "predictions_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "data": str(args.data),
        "split": args.split,
        "empty_images_scanned": len(images),
        "exported": len(rows),
        "train": sum(1 for r in rows if r["split"] == "train"),
        "val": sum(1 for r in rows if r["split"] == "val"),
        "min_models": args.min_models,
        "data_yaml": str(args.out / "data.yaml"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
