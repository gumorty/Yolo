from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO through the Ultralytics Python API.")
    parser.add_argument("--model", required=True, help="Model yaml or weight path.")
    parser.add_argument("--data", required=True, help="Dataset data.yaml path.")
    parser.add_argument("--project", required=True, help="Ultralytics output project directory.")
    parser.add_argument("--name", required=True, help="Ultralytics run name.")
    parser.add_argument("--pretrained", default="", help="Optional pretrained weight path.")
    parser.add_argument("--fallback-pretrained", default="yolov8m.pt", help="Fallback weight if pretrained is missing.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--close-mosaic", type=int, default=20)
    parser.add_argument("--mosaic", type=float, default=0.7)
    parser.add_argument("--mixup", type=float, default=0.05)
    parser.add_argument("--copy-paste", type=float, default=0.05)
    parser.add_argument("--amp", choices=["true", "false"], default="false")
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--optimizer", default="", help="Optimizer name. Set explicitly to avoid optimizer=auto overriding lr0.")
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--warmup-bias-lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--erasing", type=float)
    parser.add_argument("--auto-augment", default="")
    parser.add_argument("--multi-scale", choices=["true", "false"], default="")
    parser.add_argument("--rect", choices=["true", "false"], default="")
    parser.add_argument("--freeze", type=int)
    parser.add_argument("--fraction", type=float, help="Fraction of the dataset to train on, useful for smoke tests.")
    parser.add_argument("--seed", type=int, help="Random seed forwarded to Ultralytics.")
    parser.add_argument("--deterministic", choices=["true", "false"], default="", help="Forward deterministic mode to Ultralytics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model).expanduser()
    data_path = Path(args.data).expanduser()
    if not model_path.exists() and not str(args.model).endswith(".pt"):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing dataset yaml: {data_path}")

    model = YOLO(str(model_path))

    pretrained = Path(args.pretrained).expanduser() if args.pretrained else None
    if pretrained and pretrained.exists():
        print(f"Loading pretrained weights: {pretrained}")
        model.load(str(pretrained))
    else:
        if args.pretrained:
            print(f"Pretrained weight not found: {args.pretrained}")
        if args.fallback_pretrained:
            print(f"Loading fallback pretrained weights: {args.fallback_pretrained}")
            model.load(args.fallback_pretrained)

    train_kwargs = {
        "task": "detect",
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": str(Path(args.project).expanduser()),
        "name": args.name,
        "patience": args.patience,
        "close_mosaic": args.close_mosaic,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "workers": args.workers,
        "amp": args.amp == "true",
        "save_period": args.save_period,
    }
    if args.optimizer:
        train_kwargs["optimizer"] = args.optimizer
    auto_augment = args.auto_augment or None
    auto_augment_was_set = bool(args.auto_augment)
    if isinstance(auto_augment, str) and auto_augment.lower() in {"none", "null", "false", "off"}:
        auto_augment = None

    optional_values = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr,
        "momentum": args.momentum,
        "erasing": args.erasing,
        "freeze": args.freeze,
        "fraction": args.fraction,
        "seed": args.seed,
    }
    for key, value in optional_values.items():
        if value is not None:
            train_kwargs[key] = value
    if auto_augment_was_set:
        train_kwargs["auto_augment"] = auto_augment
    if args.multi_scale:
        train_kwargs["multi_scale"] = args.multi_scale == "true"
    if args.rect:
        train_kwargs["rect"] = args.rect == "true"
    if args.deterministic:
        train_kwargs["deterministic"] = args.deterministic == "true"

    model.train(
        **train_kwargs,
    )


if __name__ == "__main__":
    main()
