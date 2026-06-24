from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def resolve_split(data_yaml: Path, split: str) -> list[Path]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    value = data.get(split) or data.get("valid" if split == "val" else split)
    if value is None:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")
    values = value if isinstance(value, list) else [value]
    return [resolve(data_yaml.parent, str(item)) for item in values]


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def yolo_xywhn_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def area_bin(w: float, h: float) -> str:
    area = w * h
    if area < 0.001:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def read_labels(image_path: Path) -> list[dict[str, Any]]:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        return []
    labels = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        labels.append({"cls": cls, "box": yolo_xywhn_to_xyxy(x, y, w, h), "size": area_bin(w, h)})
    return labels


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iter_images(data_yaml: Path, split: str, max_images: int | None) -> list[Path]:
    images: list[Path] = []
    for image_dir in resolve_split(data_yaml, split):
        if image_dir.exists():
            images.extend(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    images = sorted(images)
    return images[:max_images] if max_images else images


def predict_all(model: Any, images: list[Path], imgsz: int, conf_min: float, iou_thr: float) -> list[dict[str, Any]]:
    cached = []
    for image_path in images:
        result = model.predict(str(image_path), imgsz=imgsz, conf=conf_min, iou=iou_thr, verbose=False)[0]
        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            h, w = result.orig_shape
            boxes = result.boxes.xyxy.cpu().tolist()
            clss = [int(v) for v in result.boxes.cls.cpu().tolist()]
            confs = [float(v) for v in result.boxes.conf.cpu().tolist()]
            for box, cls, score in zip(boxes, clss, confs):
                x1, y1, x2, y2 = box
                preds.append({"cls": cls, "conf": score, "box": (x1 / w, y1 / h, x2 / w, y2 / h)})
        cached.append({"image": str(image_path), "labels": read_labels(image_path), "preds": preds})
    return cached


def eval_threshold(cached: list[dict[str, Any]], conf: float, iou_match: float) -> dict[str, Any]:
    gt_by_size = Counter()
    hit_by_size = Counter()
    gt_by_class = Counter()
    hit_by_class = Counter()
    fp_by_class = Counter()
    total_preds = 0
    for item in cached:
        labels = item["labels"]
        preds = [p for p in item["preds"] if p["conf"] >= conf]
        total_preds += len(preds)
        for gt in labels:
            gt_by_size[gt["size"]] += 1
            gt_by_class[gt["cls"]] += 1
        used: set[int] = set()
        for gt in labels:
            best_idx = None
            best_iou = 0.0
            for idx, pred in enumerate(preds):
                if idx in used or pred["cls"] != gt["cls"]:
                    continue
                overlap = iou(gt["box"], pred["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
            if best_idx is not None and best_iou >= iou_match:
                used.add(best_idx)
                hit_by_size[gt["size"]] += 1
                hit_by_class[gt["cls"]] += 1
        for idx, pred in enumerate(preds):
            if idx not in used:
                fp_by_class[pred["cls"]] += 1

    def ratio(a: int, b: int) -> float:
        return round(a / b, 5) if b else 0.0

    total_gt = sum(gt_by_class.values())
    total_hit = sum(hit_by_class.values())
    total_fp = sum(fp_by_class.values())
    return {
        "conf": conf,
        "images": len(cached),
        "objects": total_gt,
        "predictions": total_preds,
        "matched": total_hit,
        "false_positives": total_fp,
        "fp_per_image": round(total_fp / len(cached), 5) if cached else 0.0,
        "recall": ratio(total_hit, total_gt),
        "precision_like": ratio(total_hit, total_hit + total_fp),
        "tiny_recall": ratio(hit_by_size["tiny"], gt_by_size["tiny"]),
        "small_recall": ratio(hit_by_size["small"], gt_by_size["small"]),
        "fire_recall": ratio(hit_by_class[0], gt_by_class[0]),
        "smoke_recall": ratio(hit_by_class[2], gt_by_class[2]),
        "fp_fire": fp_by_class[0],
        "fp_other": fp_by_class[1],
        "fp_smoke": fp_by_class[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan confidence thresholds for Stage6 decision making.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--conf", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    YOLO = import_yolo()
    model = YOLO(str(args.model))
    images = iter_images(args.data, args.split, args.max_images)
    cached = predict_all(model, images, args.imgsz, min(args.conf), args.iou)
    rows = [eval_threshold(cached, conf, args.iou) for conf in args.conf]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".json":
        args.out.write_text(json.dumps({"model": str(args.model), "data": str(args.data), "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        with args.out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
