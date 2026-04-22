import json
from pathlib import Path

from PIL import Image

from dataset_config import CLASS_NAMES, PROJECT_ROOT


PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def read_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.width, image.height


def parse_label_file(label_path: Path):
    boxes = []

    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label at {label_path}:{line_number}: {line}")

        x1, y1, x2, y2, class_id = map(int, parts)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid box at {label_path}:{line_number}: {line}")

        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "class_id": class_id,
            }
        )

    return boxes


def build_categories():
    return [
        {
            "id": class_id,
            "name": class_name,
            "supercategory": "pcb_defect",
        }
        for class_id, class_name in sorted(CLASS_NAMES.items())
    ]


def convert_split(split_name: str):
    image_dir = PROCESSED_ROOT / split_name / "images"
    label_dir = PROCESSED_ROOT / split_name / "labels"
    output_path = PROCESSED_ROOT / f"annotations_{split_name}.json"

    if not image_dir.exists():
        raise SystemExit(f"Image folder not found: {image_dir}")

    if not label_dir.exists():
        raise SystemExit(f"Label folder not found: {label_dir}")

    coco = {
        "info": {
            "description": "DeepPCB defect detection dataset converted to COCO format",
            "version": "1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": build_categories(),
    }

    annotation_id = 1
    image_id = 1

    for image_path in sorted(image_dir.glob("*_test.jpg")):
        image_base_id = image_path.stem.replace("_test", "")
        label_path = label_dir / f"{image_base_id}.txt"

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for image {image_path}: {label_path}")

        width, height = read_image_size(image_path)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for box in parse_label_file(label_path):
            x = box["x1"]
            y = box["y1"]
            bbox_width = box["x2"] - box["x1"]
            bbox_height = box["y2"] - box["y1"]
            area = bbox_width * bbox_height

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box["class_id"],
                    "bbox": [x, y, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            annotation_id += 1

        image_id += 1

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, indent=2)

    print(f"{split_name}:")
    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Output: {output_path}")


def main():
    for split_name in ["train", "val", "test"]:
        convert_split(split_name)

    print("\nCOCO conversion complete.")


if __name__ == "__main__":
    main()
