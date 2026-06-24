import csv
import json
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow is required to read image sizes: pip install pillow") from exc


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
HOLDOUT = PROJECT / "datasets" / "stage4_eval" / "public_hard_holdout"
IMAGES = HOLDOUT / "images"
LABELS = HOLDOUT / "labels"

MODELS = {
    "V4_P2": ROOT / "stage4" / "eval_public_holdout" / "holdout_v4_best_v2" / "predictions.json",
    "S5_FT": ROOT / "stage4" / "eval_public_holdout" / "holdout_s5_best" / "predictions.json",
    "NoP2_Abl": ROOT / "ablation_p2" / "eval_public_holdout" / "holdout_nop2_best" / "predictions.json",
}

CLASSES = {0: "fire", 2: "smoke"}
SIZE_BINS = {
    "tiny(<0.1%)": (0.0, 0.001),
    "small(0.1-1%)": (0.001, 0.01),
    "medium(1-5%)": (0.01, 0.05),
    "large(>=5%)": (0.05, float("inf")),
}
IOU_THRESHOLDS = [round(x / 100, 2) for x in range(50, 100, 5)]


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def yolo_to_xyxy(vals, width, height):
    cx, cy, bw, bh = vals
    w = bw * width
    h = bh * height
    x1 = cx * width - w / 2
    y1 = cy * height - h / 2
    return [x1, y1, x1 + w, y1 + h]


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-12)


def size_bin(area_ratio):
    for name, (lo, hi) in SIZE_BINS.items():
        if lo <= area_ratio < hi:
            return name
    raise ValueError(f"unhandled area ratio {area_ratio}")


def load_ground_truth():
    gt_by_image_class = defaultdict(list)
    counts = defaultdict(int)
    image_sizes = {}

    for image_path in sorted(IMAGES.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(image_path) as img:
            width, height = img.size
        image_id = image_path.stem
        image_sizes[image_id] = (width, height)
        label_path = LABELS / f"{image_id}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls not in CLASSES:
                continue
            box = yolo_to_xyxy([float(x) for x in parts[1:]], width, height)
            area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
            bin_name = size_bin(area_ratio)
            item = {"image_id": image_id, "cls": cls, "box": box, "size_bin": bin_name}
            gt_by_image_class[(image_id, cls)].append(item)
            counts[(CLASSES[cls], bin_name)] += 1

    return gt_by_image_class, counts


def load_predictions(path):
    preds = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        cls = int(item["category_id"]) - 1
        if cls not in CLASSES:
            continue
        preds.append(
            {
                "image_id": Path(item.get("file_name") or item["image_id"]).stem,
                "cls": cls,
                "box": xywh_to_xyxy([float(x) for x in item["bbox"]]),
                "score": float(item["score"]),
            }
        )
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def average_precision(recalls, precisions):
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for t in [x / 100 for x in range(101)]:
        vals = [p for r, p in zip(mrec, mpre) if r >= t]
        ap += max(vals) if vals else 0.0
    return ap / 101


def eval_subset(preds, gt_by_image_class, cls, bin_name, iou_thr):
    gt_pool = {}
    n_gt = 0
    for (image_id, gt_cls), items in gt_by_image_class.items():
        if gt_cls != cls:
            continue
        filtered = [g for g in items if g["size_bin"] == bin_name]
        if filtered:
            gt_pool[image_id] = filtered
            n_gt += len(filtered)
    if n_gt == 0:
        return None

    matched = {image_id: set() for image_id in gt_pool}
    tp = []
    fp = []
    for pred in [p for p in preds if p["cls"] == cls]:
        candidates = gt_pool.get(pred["image_id"], [])
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(candidates):
            if idx in matched[pred["image_id"]]:
                continue
            score = iou(pred["box"], gt["box"])
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_iou >= iou_thr and best_idx >= 0:
            matched[pred["image_id"]].add(best_idx)
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    cum_tp = []
    cum_fp = []
    stp = sfp = 0
    for t, f in zip(tp, fp):
        stp += t
        sfp += f
        cum_tp.append(stp)
        cum_fp.append(sfp)

    recalls = [x / n_gt for x in cum_tp]
    precisions = [t / max(t + f, 1e-12) for t, f in zip(cum_tp, cum_fp)]
    return {
        "ap": average_precision(recalls, precisions),
        "max_recall": max(recalls) if recalls else 0.0,
        "n_gt": n_gt,
    }


def main():
    gt_by_image_class, counts = load_ground_truth()
    rows = []
    for model_name, pred_path in MODELS.items():
        preds = load_predictions(pred_path)
        for cls, cls_name in CLASSES.items():
            for bin_name in SIZE_BINS:
                ap_values = []
                recall50 = None
                n_gt = counts.get((cls_name, bin_name), 0)
                for thr in IOU_THRESHOLDS:
                    result = eval_subset(preds, gt_by_image_class, cls, bin_name, thr)
                    if result is None:
                        continue
                    ap_values.append(result["ap"])
                    if thr == 0.5:
                        recall50 = result["max_recall"]
                rows.append(
                    {
                        "model": model_name,
                        "class": cls_name,
                        "size_bin": bin_name,
                        "gt_instances": n_gt,
                        "AP50": ap_values[0] if ap_values else "",
                        "AP50-95": sum(ap_values) / len(ap_values) if ap_values else "",
                        "Recall50": recall50 if recall50 is not None else "",
                    }
                )

    csv_path = ROOT / "per_size_ap_summary.csv"
    json_path = ROOT / "per_size_ap_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
