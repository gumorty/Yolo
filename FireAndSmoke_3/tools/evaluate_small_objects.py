from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Cannot import ultralytics.YOLO: {exc}") from exc
    return YOLO


def box_xywhn_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


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
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def area_bin(w: float, h: float) -> str:
    area = w * h
    if area < 0.001:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part.lower() == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_labels(path: Path) -> list[dict]:
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        labels.append({"cls": cls, "box": box_xywhn_to_xyxy(x, y, w, h), "bin": area_bin(w, h)})
    return labels


def resolve_split(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    value = data.get(split) or data.get("valid" if split == "val" else split)
    if value is None:
        raise FileNotFoundError(f"Split {split} not found in {data_yaml}")
    p = Path(value)
    return p if p.is_absolute() else (data_yaml.parent / p).resolve()


def evaluate(model_path: Path, data_yaml: Path, split: str, imgsz: int, conf: float, iou_thr: float, max_images: int | None) -> dict:
    YOLO = import_yolo()
    model = YOLO(str(model_path))
    image_dir = resolve_split(data_yaml, split)
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if max_images:
        images = images[:max_images]

    totals = Counter()
    matched = Counter()
    class_totals = Counter()
    class_matched = Counter()
    false_positive = Counter()

    for image_path in images:
        gts = read_labels(label_path_for(image_path))
        for gt in gts:
            totals[gt["bin"]] += 1
            class_totals[gt["cls"]] += 1

        result = model.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou_thr, verbose=False)[0]
        preds = []
        if result.boxes is not None and len(result.boxes) > 0:
            h, w = result.orig_shape
            xyxy = result.boxes.xyxy.cpu().tolist()
            cls_list = [int(v) for v in result.boxes.cls.cpu().tolist()]
            conf_list = result.boxes.conf.cpu().tolist()
            for box, cls, score in zip(xyxy, cls_list, conf_list):
                x1, y1, x2, y2 = box
                preds.append({
                    "cls": cls,
                    "score": score,
                    "box": (x1 / w, y1 / h, x2 / w, y2 / h),
                })

        used = set()
        for gt_idx, gt in enumerate(gts):
            best_idx = None
            best_iou = 0.0
            for pred_idx, pred in enumerate(preds):
                if pred_idx in used or pred["cls"] != gt["cls"]:
                    continue
                overlap = iou(gt["box"], pred["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = pred_idx
            if best_idx is not None and best_iou >= iou_thr:
                used.add(best_idx)
                matched[gt["bin"]] += 1
                class_matched[gt["cls"]] += 1

        for pred_idx, pred in enumerate(preds):
            if pred_idx not in used:
                false_positive[pred["cls"]] += 1

    recalls = {k: (matched[k] / totals[k] if totals[k] else 0.0) for k in ["tiny", "small", "medium", "large"]}
    class_recalls = {str(k): (class_matched[k] / class_totals[k] if class_totals[k] else 0.0) for k in sorted(class_totals)}
    return {
        "model": str(model_path),
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou_thr,
        "images": len(images),
        "gt_by_size": dict(totals),
        "matched_by_size": dict(matched),
        "recall_by_size": recalls,
        "gt_by_class": dict(class_totals),
        "matched_by_class": dict(class_matched),
        "recall_by_class": class_recalls,
        "false_positive_by_class": dict(false_positive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO recall by target area bin.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", default=960, type=int)
    parser.add_argument("--conf", default=0.20, type=float)
    parser.add_argument("--iou", default=0.50, type=float)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = evaluate(args.model, args.data, args.split, args.imgsz, args.conf, args.iou, args.max_images)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
