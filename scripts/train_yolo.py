import argparse
from pathlib import Path

from dataset_config import PROJECT_ROOT

try:
    from ultralytics import YOLO
except ImportError as error:
    raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from error


YOLO_ROOT = PROJECT_ROOT / "data" / "processed" / "yolo"
DATASET_YAML = YOLO_ROOT / "deeppcb.yaml"
RUNS_ROOT = PROJECT_ROOT / "models" / "yolo_runs"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DeepPCB detector with Ultralytics YOLO.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model checkpoint or preset.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size. Use 4, 2, or 1 if VRAM is tight.")
    parser.add_argument("--device", default="0", help="CUDA device index like 0, or cpu.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--project", default=str(RUNS_ROOT), help="Output directory for YOLO runs.")
    parser.add_argument("--name", default="deeppcb_yolov8", help="Run name.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DATASET_YAML.exists():
        raise SystemExit(
            f"Dataset YAML not found: {DATASET_YAML}\n"
            "Run: python scripts/convert_deeppcb_to_yolo.py"
        )

    Path(args.project).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
        cache=False,
        amp=False,
        patience=10,
        plots=True,
    )


if __name__ == "__main__":
    main()
