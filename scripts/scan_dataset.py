from collections import Counter
from pathlib import Path

from dataset_config import CLASS_NAMES, RAW_DATASET_ROOT


def annotation_path_for_image(image_path: Path) -> Path:
    base_id = image_path.stem.replace("_test", "")
    annotation_dir = Path(str(image_path.parent) + "_not")
    return annotation_dir / f"{base_id}.txt"


def parse_annotation_file(path: Path):
    boxes = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid annotation at {path}:{line_number}: {line}")
        x1, y1, x2, y2, class_id = map(int, parts)
        boxes.append((x1, y1, x2, y2, class_id))
    return boxes


def main():
    if not RAW_DATASET_ROOT.exists():
        raise SystemExit(f"Dataset not found: {RAW_DATASET_ROOT}")

    jpg_files = list(RAW_DATASET_ROOT.rglob("*.jpg"))
    test_images = sorted(RAW_DATASET_ROOT.rglob("*_test.jpg"))
    temp_images = sorted(RAW_DATASET_ROOT.rglob("*_temp.jpg"))
    txt_files = sorted(RAW_DATASET_ROOT.rglob("*.txt"))

    missing_annotations = []
    class_counts = Counter()
    boxes_per_image = []

    for image_path in test_images:
        annotation_path = annotation_path_for_image(image_path)
        if not annotation_path.exists():
            missing_annotations.append(image_path)
            continue

        boxes = parse_annotation_file(annotation_path)
        boxes_per_image.append(len(boxes))
        class_counts.update(box[-1] for box in boxes)

    print("DeepPCB Dataset Scan")
    print("====================")
    print(f"Dataset root: {RAW_DATASET_ROOT}")
    print(f"JPG files: {len(jpg_files)}")
    print(f"Template images: {len(temp_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Annotation files: {len(txt_files)}")
    print(f"Annotated test images: {len(test_images) - len(missing_annotations)}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Total defect boxes: {sum(class_counts.values())}")

    if boxes_per_image:
        average = sum(boxes_per_image) / len(boxes_per_image)
        print(f"Boxes per annotated image: min={min(boxes_per_image)}, avg={average:.2f}, max={max(boxes_per_image)}")

    print("\nClass counts")
    for class_id in sorted(CLASS_NAMES):
        print(f"{class_id}: {CLASS_NAMES[class_id]} = {class_counts[class_id]}")

    if missing_annotations:
        print("\nFirst missing annotations")
        for image_path in missing_annotations[:10]:
            print(image_path)


if __name__ == "__main__":
    main()

