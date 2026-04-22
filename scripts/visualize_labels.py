import argparse
from pathlib import Path

import cv2

from dataset_config import CLASS_NAMES, RAW_DATASET_ROOT, VISUALIZATION_ROOT
from scan_dataset import annotation_path_for_image, parse_annotation_file


COLORS = {
    1: (64, 64, 255),
    2: (0, 180, 255),
    3: (0, 220, 0),
    4: (255, 170, 0),
    5: (255, 80, 180),
    6: (180, 80, 255),
}


def draw_annotations(image_path: Path, output_path: Path) -> bool:
    annotation_path = annotation_path_for_image(image_path)
    if not annotation_path.exists():
        return False

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for x1, y1, x2, y2, class_id in parse_annotation_file(annotation_path):
        color = COLORS.get(class_id, (255, 255, 255))
        label = CLASS_NAMES.get(class_id, str(class_id))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def main():
    parser = argparse.ArgumentParser(description="Draw DeepPCB ground-truth boxes on sample images.")
    parser.add_argument("--limit", type=int, default=20, help="Number of labeled images to visualize.")
    args = parser.parse_args()

    written = 0
    for image_path in sorted(RAW_DATASET_ROOT.rglob("*_test.jpg")):
        output_path = VISUALIZATION_ROOT / f"{image_path.stem}_labels.jpg"
        if draw_annotations(image_path, output_path):
            written += 1
            print(f"Wrote {output_path}")
        if written >= args.limit:
            break

    print(f"\nDone. Created {written} visualization images in {VISUALIZATION_ROOT}")


if __name__ == "__main__":
    main()

