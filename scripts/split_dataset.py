from pathlib import Path
from shutil import copy2
import random

from dataset_config import RAW_DATASET_ROOT, PROJECT_ROOT
from scan_dataset import annotation_path_for_image


RANDOM_SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def collect_labeled_images():
    labeled_images = []

    for image_path in sorted(RAW_DATASET_ROOT.rglob("*_test.jpg")):
        annotation_path = annotation_path_for_image(image_path)

        if annotation_path.exists():
            labeled_images.append((image_path, annotation_path))

    return labeled_images


def prepare_output_folders():
    for split_name in ["train", "val", "test"]:
        (PROCESSED_ROOT / split_name / "images").mkdir(parents=True, exist_ok=True)
        (PROCESSED_ROOT / split_name / "labels").mkdir(parents=True, exist_ok=True)


def copy_split_files(split_name, samples):
    image_output_dir = PROCESSED_ROOT / split_name / "images"
    label_output_dir = PROCESSED_ROOT / split_name / "labels"

    for image_path, annotation_path in samples:
        copy2(image_path, image_output_dir / image_path.name)
        copy2(annotation_path, label_output_dir / annotation_path.name)


def write_split_file(split_name, samples):
    split_file = PROCESSED_ROOT / f"{split_name}.txt"

    with split_file.open("w", encoding="utf-8") as file:
        for image_path, _ in samples:
            file.write(f"{image_path.name}\n")


def main():
    if not RAW_DATASET_ROOT.exists():
        raise SystemExit(f"Dataset not found: {RAW_DATASET_ROOT}")

    labeled_images = collect_labeled_images()

    if not labeled_images:
        raise SystemExit("No labeled images found.")

    random.seed(RANDOM_SEED)
    random.shuffle(labeled_images)

    total_count = len(labeled_images)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)

    train_samples = labeled_images[:train_count]
    val_samples = labeled_images[train_count:train_count + val_count]
    test_samples = labeled_images[train_count + val_count:]

    prepare_output_folders()

    copy_split_files("train", train_samples)
    copy_split_files("val", val_samples)
    copy_split_files("test", test_samples)

    write_split_file("train", train_samples)
    write_split_file("val", val_samples)
    write_split_file("test", test_samples)

    print("Dataset split complete")
    print("======================")
    print(f"Total labeled images: {total_count}")
    print(f"Train: {len(train_samples)}")
    print(f"Validation: {len(val_samples)}")
    print(f"Test: {len(test_samples)}")
    print(f"Output folder: {PROCESSED_ROOT}")


if __name__ == "__main__":
    main()
