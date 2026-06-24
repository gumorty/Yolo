from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from check_stage4_readiness import highres_ratio, count_class_size
from dataset_audit import audit_dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_relative_dataset_yaml(data_yaml: Path) -> bool:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    return all(not Path(str(data[key])).is_absolute() for key in ("train", "val", "test") if key in data)


def count_images(image_dir: Path) -> int:
    return sum(1 for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def resolve_split(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    value = data[split]
    path = Path(str(value))
    return path if path.is_absolute() else (data_yaml.parent / path).resolve()


def model_load_check(model_yaml: Path, pretrained: Path) -> dict[str, Any]:
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_yaml))
        model.load(str(pretrained))
        return {"pass": True, "message": "Model YAML builds and pretrained weights load."}
    except Exception as exc:
        return {"pass": False, "message": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage4 training preflight check.")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    project = args.project.resolve()
    data_yaml = args.data or project / "datasets" / "stage4_full_tile_sensors3" / "data.yaml"
    holdout_yaml = args.holdout or project / "datasets" / "stage4_eval" / "public_hard_holdout" / "data.yaml"
    model_yaml = args.model or project / "models" / "yolov8m-p2-fire-smoke-3cls.yaml"
    pretrained = args.pretrained or project / "weights" / "sensors_yolov8m_3cls_100_best.pt"
    out = args.out or project / "reports" / "stage4_preflight_check.json"

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    add("data_yaml_exists", data_yaml.exists(), str(data_yaml))
    add("holdout_yaml_exists", holdout_yaml.exists(), str(holdout_yaml))
    add("model_yaml_exists", model_yaml.exists(), str(model_yaml))
    add("pretrained_exists", pretrained.exists(), str(pretrained))

    if data_yaml.exists():
        add("data_yaml_uses_relative_paths", is_relative_dataset_yaml(data_yaml), yaml.safe_load(data_yaml.read_text(encoding="utf-8")))
        train_dir = resolve_split(data_yaml, "train")
        val_dir = resolve_split(data_yaml, "val")
        add("train_images_count", count_images(train_dir) > 0, count_images(train_dir))
        add("val_images_count", count_images(val_dir) > 0, count_images(val_dir))
        audit = audit_dataset(data_yaml)
        readiness_checks = {
            "train_tiny_small_fire": count_class_size(audit, "fire", ("tiny", "small"), "train"),
            "train_tiny_small_smoke": count_class_size(audit, "smoke", ("tiny", "small"), "train"),
            "valid_tiny_small_fire": count_class_size(audit, "fire", ("tiny", "small"), "valid"),
            "valid_tiny_small_smoke": count_class_size(audit, "smoke", ("tiny", "small"), "valid"),
            "highres_ratio": highres_ratio(audit),
        }
        add(
            "readiness_minimums",
            readiness_checks["train_tiny_small_fire"] >= 5000
            and readiness_checks["train_tiny_small_smoke"] >= 1000
            and readiness_checks["valid_tiny_small_smoke"] >= 100
            and readiness_checks["highres_ratio"] >= 0.20,
            readiness_checks,
        )

    if holdout_yaml.exists():
        add("holdout_yaml_uses_relative_paths", is_relative_dataset_yaml(holdout_yaml), yaml.safe_load(holdout_yaml.read_text(encoding="utf-8")))
        holdout_dir = resolve_split(holdout_yaml, "val")
        add("holdout_images_count", count_images(holdout_dir) >= 300, count_images(holdout_dir))

    if model_yaml.exists() and pretrained.exists():
        load = model_load_check(model_yaml, pretrained)
        add("model_pretrained_load", load["pass"], load["message"])

    report = {
        "pass": all(item["pass"] for item in checks),
        "checks": checks,
        "recommendation": "Ready for Stage4 scout training." if all(item["pass"] for item in checks) else "Do not train yet; fix failed checks.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
