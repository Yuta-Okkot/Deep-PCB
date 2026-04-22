from pathlib import Path
from shutil import move

from dataset_config import PROJECT_ROOT, RAW_DATASET_ROOT
from scan_dataset import annotation_path_for_image


EXCLUDED_ROOT = PROJECT_ROOT / "data" / "excluded_missing_labels"


def relative_group_path(path: Path) -> Path:
    return path.relative_to(RAW_DATASET_ROOT)


def move_if_exists(source: Path) -> Path | None:
    if not source.exists():
        return None

    destination = EXCLUDED_ROOT / relative_group_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    move(str(source), str(destination))
    return destination


def main():
    if not RAW_DATASET_ROOT.exists():
        raise SystemExit(f"Dataset not found: {RAW_DATASET_ROOT}")

    missing_label_images = []
    for image_path in sorted(RAW_DATASET_ROOT.rglob("*_test.jpg")):
        if not annotation_path_for_image(image_path).exists():
            missing_label_images.append(image_path)

    print(f"Found {len(missing_label_images)} test images without labels.")

    moved_files = []
    for test_image in missing_label_images:
        temp_image = test_image.with_name(test_image.name.replace("_test.jpg", "_temp.jpg"))

        for source in (test_image, temp_image):
            destination = move_if_exists(source)
            if destination is not None:
                moved_files.append((source, destination))
                print(f"Moved {source} -> {destination}")

    print(f"\nDone. Moved {len(moved_files)} files to {EXCLUDED_ROOT}")


if __name__ == "__main__":
    main()

