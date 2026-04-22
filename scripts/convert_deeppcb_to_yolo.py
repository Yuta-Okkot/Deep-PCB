from pathlib import Path
from shutil import copyfile

from PIL import Image

from dataset_config import CLASS_NAMES, PROJECT_ROOT


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
YOLO_ROOT = PROCESSED_ROOT / "yolo"


def ensure_dirs():
    for split_name in ["train", "val", "test"]:
        (YOLO_ROOT / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def convert_box_to_yolo(x1: float, y1: float, x2: float, y2: float, width: int, height: int):
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0

    return (
        center_x / width,
        center_y / height,
        box_width / width,
        box_height / height,
    )


def convert_split(split_name: str):
    source_image_dir = PROCESSED_ROOT / split_name / "images"
    source_label_dir = PROCESSED_ROOT / split_name / "labels"
    target_image_dir = YOLO_ROOT / "images" / split_name
    target_label_dir = YOLO_ROOT / "labels" / split_name

    written_labels = 0
    written_boxes = 0

    for image_path in sorted(source_image_dir.glob("*_test.jpg")):
        label_path = source_label_dir / f"{image_path.stem.replace('_test', '')}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}: {label_path}")

        with Image.open(image_path) as image:
            width, height = image.size

        target_image_path = target_image_dir / image_path.name
        target_label_path = target_label_dir / f"{image_path.stem}.txt"

        copyfile(image_path, target_image_path)

        yolo_lines = []
        for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}: {line}")

            x1, y1, x2, y2, class_id = map(float, parts)
            class_id = int(class_id) - 1

            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"Invalid class id at {label_path}:{line_number}: {line}")

            cx, cy, bw, bh = convert_box_to_yolo(x1, y1, x2, y2, width, height)
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            written_boxes += 1

        target_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        written_labels += 1

    print(f"{split_name}: images={written_labels}, boxes={written_boxes}")


def write_dataset_yaml():
    yaml_path = YOLO_ROOT / "deeppcb.yaml"
    names = [CLASS_NAMES[index] for index in sorted(CLASS_NAMES)]

    yaml_lines = [
        f"path: {YOLO_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"Wrote {yaml_path}")


def main():
    ensure_dirs()
    for split_name in ["train", "val", "test"]:
        convert_split(split_name)
    write_dataset_yaml()
    print("\nYOLO conversion complete.")


if __name__ == "__main__":
    main()
