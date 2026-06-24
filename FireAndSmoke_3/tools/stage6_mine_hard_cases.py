from __future__ import annotations

import argparse
import csv
import json
import shutil
from fnmatch import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
CLASS_NAMES = {0: "fire", 1: "other", 2: "smoke"}


@dataclass(frozen=True)
class Prediction:
    cls: int
    conf: float
    xyxy: tuple[float, float, float, float]

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.cls, str(self.cls))


def import_yolo():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit(f"Cannot import ultralytics.YOLO: {exc}") from exc
    return YOLO


def is_skipped(path: Path, patterns: list[str]) -> bool:
    name = path.name.lower()
    return any(fnmatch(name, pattern.lower()) for pattern in patterns)


def iter_sources(paths: list[Path], skip_patterns: list[str]) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    videos: list[Path] = []
    for source in paths:
        if source.is_file():
            suffix = source.suffix.lower()
            if is_skipped(source, skip_patterns):
                continue
            if suffix in IMAGE_EXTS:
                images.append(source)
            elif suffix in VIDEO_EXTS:
                videos.append(source)
            continue
        if source.is_dir():
            images.extend(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS and not is_skipped(p, skip_patterns))
            videos.extend(p for p in source.rglob("*") if p.suffix.lower() in VIDEO_EXTS and not is_skipped(p, skip_patterns))
    return sorted(set(images)), sorted(set(videos))


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part.lower() == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def yolo_to_xyxy(row: list[str], width: int, height: int) -> tuple[int, tuple[float, float, float, float]] | None:
    if len(row) < 5:
        return None
    try:
        cls = int(float(row[0]))
        x, y, w, h = [float(v) for v in row[1:5]]
    except ValueError:
        return None
    return cls, ((x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height)


def read_labels(image_path: Path, width: int, height: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        return []
    labels: list[tuple[int, tuple[float, float, float, float]]] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = yolo_to_xyxy(raw.strip().split(), width, height)
        if parsed is not None:
            labels.append(parsed)
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


def predict_frame(model: Any, frame: Any, imgsz: int, conf: float, iou_thr: float) -> list[Prediction]:
    result = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou_thr, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().tolist()
    clss = [int(v) for v in result.boxes.cls.cpu().tolist()]
    confs = [float(v) for v in result.boxes.conf.cpu().tolist()]
    return [Prediction(cls=cls, conf=score, xyxy=tuple(box)) for box, cls, score in zip(boxes, clss, confs)]


def classify_image_case(
    preds: list[Prediction],
    labels: list[tuple[int, tuple[float, float, float, float]]],
    match_iou: float,
) -> tuple[str, int, int, int]:
    if not labels:
        return ("empty_with_detection" if preds else "empty_clean", len(preds), 0, 0)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    wrong_class = 0
    for gt_idx, (gt_cls, gt_box) in enumerate(labels):
        best_idx = None
        best_iou = 0.0
        best_same_class = False
        for pred_idx, pred in enumerate(preds):
            if pred_idx in matched_pred:
                continue
            overlap = iou(gt_box, pred.xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = pred_idx
                best_same_class = pred.cls == gt_cls
        if best_idx is not None and best_iou >= match_iou and best_same_class:
            matched_gt.add(gt_idx)
            matched_pred.add(best_idx)
        elif best_idx is not None and best_iou >= match_iou:
            wrong_class += 1
    false_positive = len(preds) - len(matched_pred)
    missed = len(labels) - len(matched_gt)
    if false_positive and missed:
        return "mixed_fp_miss", false_positive, missed, wrong_class
    if false_positive:
        return "false_positive", false_positive, missed, wrong_class
    if missed:
        return "missed_or_incomplete", false_positive, missed, wrong_class
    return "matched", false_positive, missed, wrong_class


def draw_predictions(frame: Any, preds: list[Prediction], case_type: str) -> Any:
    canvas = frame.copy()
    colors = {0: (0, 80, 255), 1: (255, 180, 0), 2: (180, 180, 180)}
    for pred in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in pred.xyxy]
        color = colors.get(pred.cls, (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{pred.class_name} {pred.conf:.2f}"
        cv2.putText(canvas, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(canvas, case_type, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
    cv2.putText(canvas, case_type, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
    return canvas


def safe_stem(path: Path, frame_idx: int | None = None) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    if frame_idx is not None:
        return f"{stem}_f{frame_idx:06d}"
    return stem


def save_case(
    frame: Any,
    source: Path,
    stem: str,
    preds: list[Prediction],
    case_type: str,
    out_dir: Path,
    save_crops: bool,
) -> dict[str, Any]:
    review_img = out_dir / "review_images" / f"{stem}.jpg"
    annotated_img = out_dir / "annotated" / f"{stem}.jpg"
    review_img.parent.mkdir(parents=True, exist_ok=True)
    annotated_img.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(review_img), frame)
    cv2.imwrite(str(annotated_img), draw_predictions(frame, preds, case_type))

    crop_paths: list[str] = []
    if save_crops:
        crop_dir = out_dir / "crops" / case_type
        crop_dir.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        for idx, pred in enumerate(preds):
            x1, y1, x2, y2 = [int(round(v)) for v in pred.xyxy]
            pad = 16
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_path = crop_dir / f"{stem}_p{idx:02d}_{pred.class_name}_{pred.conf:.2f}.jpg"
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            crop_paths.append(str(crop_path))

    return {
        "source": str(source),
        "review_image": str(review_img),
        "annotated_image": str(annotated_img),
        "case_type": case_type,
        "predictions": [
            {
                "class_id": pred.cls,
                "class_name": pred.class_name,
                "confidence": round(pred.conf, 5),
                "xyxy": [round(v, 2) for v in pred.xyxy],
            }
            for pred in preds
        ],
        "crops": crop_paths,
        "review_decision": "pending",
        "notes": "",
    }


def mine_images(model: Any, images: list[Path], args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = images[: args.max_images] if args.max_images else images
    for image_path in selected:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        preds = predict_frame(model, frame, args.imgsz, args.conf, args.iou)
        labels = read_labels(image_path, width, height)
        case_type, fp_count, missed_count, wrong_class = classify_image_case(preds, labels, args.match_iou)
        if case_type in {"matched", "empty_clean"} and not args.keep_clean:
            continue
        row = save_case(frame, image_path, safe_stem(image_path), preds, case_type, out_dir, args.save_crops)
        row.update(
            {
                "source_type": "image",
                "frame_index": "",
                "label_count": len(labels),
                "prediction_count": len(preds),
                "false_positive_count": fp_count,
                "missed_count": missed_count,
                "wrong_class_count": wrong_class,
            }
        )
        rows.append(row)
    return rows


def mine_videos(model: Any, videos: list[Path], args: argparse.Namespace, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue
        saved_for_video = 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.video_every == 0:
                preds = predict_frame(model, frame, args.imgsz, args.conf, args.iou)
                if preds or args.keep_clean:
                    case_type = "video_detection_needs_review" if preds else "video_empty_sample"
                    row = save_case(frame, video_path, safe_stem(video_path, frame_idx), preds, case_type, out_dir, args.save_crops)
                    row.update(
                        {
                            "source_type": "video",
                            "frame_index": frame_idx,
                            "label_count": "",
                            "prediction_count": len(preds),
                            "false_positive_count": "",
                            "missed_count": "",
                            "wrong_class_count": "",
                        }
                    )
                    rows.append(row)
                    saved_for_video += 1
                    if args.max_frames_per_video and saved_for_video >= args.max_frames_per_video:
                        break
            frame_idx += 1
        cap.release()
    return rows


def write_review_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "review_manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "review_manifest.csv"
    fieldnames = [
        "source_type",
        "source",
        "frame_index",
        "case_type",
        "review_image",
        "annotated_image",
        "label_count",
        "prediction_count",
        "false_positive_count",
        "missed_count",
        "wrong_class_count",
        "review_decision",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "total_cases": len(rows),
        "case_counts": {},
        "review_manifest_jsonl": str(jsonl_path),
        "review_manifest_csv": str(csv_path),
    }
    for row in rows:
        case_type = str(row.get("case_type", "unknown"))
        summary["case_counts"][case_type] = summary["case_counts"].get(case_type, 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_empty_label_dataset(rows: list[dict[str, Any]], out_dir: Path) -> None:
    dataset_dir = out_dir / "candidate_empty_yolo"
    image_dir = dataset_dir / "train" / "images"
    label_dir = dataset_dir / "train" / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        if row.get("review_decision") != "confirmed_empty":
            continue
        src = Path(str(row["review_image"]))
        dst = image_dir / f"hardneg_{idx:06d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        (label_dir / f"{dst.stem}.txt").write_text("", encoding="utf-8")
    (dataset_dir / "data.yaml").write_text(
        "train: train/images\nval: train/images\nnc: 3\nnames:\n- fire\n- other\n- smoke\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Stage6 false-positive, missed, and video review cases.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--video-every", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-frames-per-video", type=int, default=80)
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--keep-clean", action="store_true", help="Also save clean empty/matched samples for audit.")
    parser.add_argument(
        "--skip-pattern",
        action="append",
        default=["result_*", "annotated_*", "corrected_*", "final_*", "*_annotated.*"],
        help="Filename glob to skip. Can be repeated.",
    )
    parser.add_argument("--write-empty-label-dataset", action="store_true", help="Only uses rows manually changed to review_decision=confirmed_empty.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    YOLO = import_yolo()
    model = YOLO(str(args.model))
    images, videos = iter_sources(args.source, args.skip_pattern)
    rows = []
    rows.extend(mine_images(model, images, args, args.out))
    rows.extend(mine_videos(model, videos, args, args.out))
    write_review_outputs(rows, args.out)
    if args.write_empty_label_dataset:
        write_empty_label_dataset(rows, args.out)
    print(json.dumps({"images": len(images), "videos": len(videos), "saved_cases": len(rows), "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
