from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "fire", 1: "other", 2: "smoke"}


@dataclass(frozen=True)
class EvalConfig:
    name: str
    model: Path
    imgsz: int
    conf: float
    iou: float
    mode: str = "normal"
    slice_size: int = 960
    overlap: float = 0.25


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Cannot import ultralytics.YOLO: {exc}") from exc
    return YOLO


def area_bin(w: float, h: float) -> str:
    area = w * h
    if area < 0.001:
        return "tiny"
    if area < 0.01:
        return "small"
    if area < 0.05:
        return "medium"
    return "large"


def yolo_xywhn_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
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


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_labels(image_path: Path) -> list[dict[str, Any]]:
    label_path = label_path_for(image_path)
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        x, y, w, h = [float(v) for v in parts[1:5]]
        labels.append(
            {
                "cls": cls,
                "class_name": CLASS_NAMES.get(cls, str(cls)),
                "box": yolo_xywhn_to_xyxy(x, y, w, h),
                "size": area_bin(w, h),
            }
        )
    return labels


def iter_slices(width: int, height: int, slice_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    slice_size = max(320, min(int(slice_size), max(width, height)))
    stride = max(160, int(slice_size * (1.0 - overlap)))
    xs = list(range(0, max(width - slice_size, 0) + 1, stride))
    ys = list(range(0, max(height - slice_size, 0) + 1, stride))
    if not xs or xs[-1] != max(width - slice_size, 0):
        xs.append(max(width - slice_size, 0))
    if not ys or ys[-1] != max(height - slice_size, 0):
        ys.append(max(height - slice_size, 0))
    return [(x, y, min(x + slice_size, width), min(y + slice_size, height)) for y in ys for x in xs]


def result_preds(result: Any, offset_x: int = 0, offset_y: int = 0) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return preds
    orig_h, orig_w = result.orig_shape
    xyxy = result.boxes.xyxy.cpu().tolist()
    cls_list = [int(v) for v in result.boxes.cls.cpu().tolist()]
    conf_list = result.boxes.conf.cpu().tolist()
    for box, cls, score in zip(xyxy, cls_list, conf_list):
        x1, y1, x2, y2 = box
        preds.append(
            {
                "cls": cls,
                "class_name": CLASS_NAMES.get(cls, str(cls)),
                "score": float(score),
                "box": (
                    (x1 + offset_x) / orig_w if offset_x == 0 else x1 + offset_x,
                    (y1 + offset_y) / orig_h if offset_y == 0 else y1 + offset_y,
                    (x2 + offset_x) / orig_w if offset_x == 0 else x2 + offset_x,
                    (y2 + offset_y) / orig_h if offset_y == 0 else y2 + offset_y,
                ),
            }
        )
    return preds


def merge_preds(preds: list[dict[str, Any]], iou_thr: float) -> list[dict[str, Any]]:
    ordered = sorted(preds, key=lambda p: p["score"], reverse=True)
    kept: list[dict[str, Any]] = []
    for pred in ordered:
        if any(pred["cls"] == old["cls"] and box_iou(pred["box"], old["box"]) >= iou_thr for old in kept):
            continue
        kept.append(pred)
    return kept


def predict(model: Any, image_path: Path, cfg: EvalConfig) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    if cfg.mode == "normal":
        result = model.predict(str(image_path), imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False)[0]
        return result_preds(result), time.perf_counter() - start

    image = cv2.imread(str(image_path))
    if image is None:
        return [], time.perf_counter() - start
    height, width = image.shape[:2]
    all_preds: list[dict[str, Any]] = []
    if cfg.mode == "full_sliced":
        result = model.predict(image, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False)[0]
        all_preds.extend(result_preds(result))
    for x1, y1, x2, y2 in iter_slices(width, height, cfg.slice_size, cfg.overlap):
        crop = image[y1:y2, x1:x2]
        result = model.predict(crop, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, verbose=False)[0]
        crop_preds = result_preds(result)
        for pred in crop_preds:
            px1, py1, px2, py2 = pred["box"]
            pred["box"] = ((px1 * (x2 - x1) + x1) / width, (py1 * (y2 - y1) + y1) / height, (px2 * (x2 - x1) + x1) / width, (py2 * (y2 - y1) + y1) / height)
        all_preds.extend(crop_preds)
    return merge_preds(all_preds, cfg.iou), time.perf_counter() - start


def choose_images(image_dir: Path, max_images: int, seed: int, require_small_fire_smoke: bool) -> list[Path]:
    images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if require_small_fire_smoke:
        selected = []
        for image in images:
            labels = read_labels(image)
            if any(item["class_name"] in {"fire", "smoke"} and item["size"] in {"tiny", "small"} for item in labels):
                selected.append(image)
        images = selected
    rng = random.Random(seed)
    rng.shuffle(images)
    return images[:max_images]


def evaluate_config(cfg: EvalConfig, images: list[Path]) -> dict[str, Any]:
    YOLO = import_yolo()
    model = YOLO(str(cfg.model))
    gt_by_size: Counter[str] = Counter()
    hit_by_size: Counter[str] = Counter()
    gt_by_class: Counter[str] = Counter()
    hit_by_class: Counter[str] = Counter()
    fp_by_class: Counter[str] = Counter()
    miss_examples: list[dict[str, Any]] = []
    total_pred = 0
    elapsed = 0.0

    for image in images:
        gts = read_labels(image)
        for gt in gts:
            gt_by_size[gt["size"]] += 1
            gt_by_class[gt["class_name"]] += 1

        preds, seconds = predict(model, image, cfg)
        elapsed += seconds
        total_pred += len(preds)

        used: set[int] = set()
        missed: list[dict[str, Any]] = []
        for gt_idx, gt in enumerate(gts):
            best_idx = None
            best_iou = 0.0
            for pred_idx, pred in enumerate(preds):
                if pred_idx in used or pred["cls"] != gt["cls"]:
                    continue
                overlap = box_iou(gt["box"], pred["box"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = pred_idx
            if best_idx is not None and best_iou >= cfg.iou:
                used.add(best_idx)
                hit_by_size[gt["size"]] += 1
                hit_by_class[gt["class_name"]] += 1
            else:
                missed.append({"class_name": gt["class_name"], "size": gt["size"], "best_iou": round(best_iou, 4)})

        for pred_idx, pred in enumerate(preds):
            if pred_idx not in used:
                fp_by_class[pred["class_name"]] += 1

        if missed and len(miss_examples) < 30:
            miss_examples.append({"image": str(image), "missed": missed[:5], "pred_count": len(preds)})

    def ratio(hit: int, total: int) -> float:
        return round(hit / total, 4) if total else 0.0

    total_gt = sum(gt_by_class.values())
    total_hit = sum(hit_by_class.values())
    total_fp = sum(fp_by_class.values())
    return {
        "name": cfg.name,
        "model": str(cfg.model),
        "imgsz": cfg.imgsz,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "mode": cfg.mode,
        "slice_size": cfg.slice_size,
        "overlap": cfg.overlap,
        "images": len(images),
        "objects": total_gt,
        "predictions": total_pred,
        "matched": total_hit,
        "false_positives": total_fp,
        "recall": ratio(total_hit, total_gt),
        "precision_like": ratio(total_hit, total_hit + total_fp),
        "seconds": round(elapsed, 3),
        "fps": round(len(images) / elapsed, 3) if elapsed else 0.0,
        "gt_by_size": dict(gt_by_size),
        "recall_by_size": {key: ratio(hit_by_size[key], gt_by_size[key]) for key in ["tiny", "small", "medium", "large"]},
        "gt_by_class": dict(gt_by_class),
        "recall_by_class": {key: ratio(hit_by_class[key], gt_by_class[key]) for key in ["fire", "other", "smoke"]},
        "false_positive_by_class": dict(fp_by_class),
        "miss_examples": miss_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--old-model", required=True, type=Path)
    parser.add_argument("--new-model", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-small-fire-smoke", action="store_true")
    parser.add_argument("--include-sliced", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    images = choose_images(args.image_dir, args.max_images, args.seed, args.require_small_fire_smoke)
    configs = [
        EvalConfig("old_yolov8m_3cls_normal_640", args.old_model, 640, 0.20, 0.50),
        EvalConfig("new_p2_normal_960", args.new_model, 960, 0.20, 0.50),
    ]
    if args.include_sliced:
        configs.append(EvalConfig("new_p2_full_sliced_960_slice640", args.new_model, 960, 0.20, 0.50, "full_sliced", 640, 0.30))

    report = {
        "image_dir": str(args.image_dir),
        "max_images": args.max_images,
        "seed": args.seed,
        "require_small_fire_smoke": args.require_small_fire_smoke,
        "evaluated_images": [str(p) for p in images],
        "results": [evaluate_config(cfg, images) for cfg in configs],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
