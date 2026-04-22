import argparse
import json
from pathlib import Path

import tensorflow as tf

try:
    import keras_cv
except ImportError as error:
    raise SystemExit(
        "keras-cv is not installed. Run: pip install -r requirements.txt"
    ) from error

from dataset_config import CLASS_NAMES, PROJECT_ROOT


MAX_BOXES = 32
NUM_CLASSES = len(CLASS_NAMES)
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
MODEL_ROOT = PROJECT_ROOT / "models"
CHECKPOINT_ROOT = MODEL_ROOT / "checkpoints"
SAVED_MODEL_PATH = MODEL_ROOT / "pcb_yolov8_detector.keras"


CLASS_MAPPING = {
    index - 1: name
    for index, name in CLASS_NAMES.items()
}


def load_coco_records(split_name: str):
    annotation_path = PROCESSED_ROOT / f"annotations_{split_name}.json"
    image_dir = PROCESSED_ROOT / split_name / "images"

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    images_by_id = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image_id = {
        image_id: []
        for image_id in images_by_id
    }

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        x, y, width, height = annotation["bbox"]

        annotations_by_image_id[image_id].append(
            {
                "box": [x, y, x + width, y + height],
                "class_id": annotation["category_id"] - 1,
            }
        )

    records = []
    for image_id, image in sorted(images_by_id.items()):
        image_path = image_dir / image["file_name"]
        annotations = annotations_by_image_id[image_id]

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        records.append(
            {
                "image_path": str(image_path),
                "boxes": [item["box"] for item in annotations],
                "classes": [item["class_id"] for item in annotations],
            }
        )

    return records


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def load_sample(image_path, boxes, classes):
    image = load_image(image_path)

    bounding_boxes = {
        "boxes": tf.cast(boxes, tf.float32),
        "classes": tf.cast(classes, tf.float32),
    }

    return {
        "images": image,
        "bounding_boxes": bounding_boxes,
    }


def dict_to_tuple(sample):
    return sample["images"], sample["bounding_boxes"]


def densify_bounding_boxes(sample):
    sample["bounding_boxes"] = keras_cv.bounding_box.to_dense(
        sample["bounding_boxes"],
        max_boxes=MAX_BOXES,
    )
    return sample


def make_dataset(split_name: str, batch_size: int, training: bool, image_size: int):
    records = load_coco_records(split_name)

    image_paths = [record["image_path"] for record in records]
    boxes = [record["boxes"] for record in records]
    classes = [record["classes"] for record in records]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(image_paths),
            tf.ragged.constant(boxes, dtype=tf.float32),
            tf.ragged.constant(classes, dtype=tf.float32),
        )
    )

    dataset = dataset.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        dataset = dataset.shuffle(batch_size * 8, reshuffle_each_iteration=True)

    dataset = dataset.ragged_batch(batch_size, drop_remainder=training)

    resize = keras_cv.layers.JitteredResize(
        target_size=(image_size, image_size),
        scale_factor=(1.0, 1.0),
        bounding_box_format="xyxy",
    )
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(densify_bounding_boxes, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(records)


class EvaluateCOCOMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, save_path: Path):
        super().__init__()
        self.validation_data_for_metrics = validation_data
        self.save_path = save_path
        self.best_map = -1.0
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format="xyxy",
            evaluate_freq=1e9,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.metrics.reset_state()

        for images, y_true in self.validation_data_for_metrics:
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metric_results = self.metrics.result(force=True)
        logs.update(metric_results)

        current_map = float(metric_results.get("MaP", -1.0))
        print(f"\nValidation MaP: {current_map:.4f}")

        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)
            print(f"Saved best model to {self.save_path}")


def build_model(learning_rate: float):
    backbone = keras_cv.models.YOLOV8Backbone.from_preset(
        "yolo_v8_xs_backbone_coco"
    )

    model = keras_cv.models.YOLOV8Detector(
        num_classes=NUM_CLASSES,
        bounding_box_format="xyxy",
        backbone=backbone,
        fpn_depth=1,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        global_clipnorm=10.0,
    )

    model.compile(
        optimizer=optimizer,
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        jit_compile=False,
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PCB fault detector with KerasCV YOLOv8.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size. Use 2 for CPU or low RAM.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=416,
        help="Square training image size. Use 416 or 320 for older 6 GB GPUs.",
    )
    parser.add_argument(
        "--skip-coco-metrics",
        action="store_true",
        help="Skip mAP callback to make a quick smoke test faster.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Older GPUs under WSL can fail on XLA/cuDNN autotuned kernels.
    tf.config.optimizer.set_jit(False)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    train_ds, train_count = make_dataset("train", args.batch_size, training=True, image_size=args.image_size)
    val_ds, val_count = make_dataset("val", args.batch_size, training=False, image_size=args.image_size)

    print("Training setup")
    print("==============")
    print(f"Train images: {train_count}")
    print(f"Validation images: {val_count}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Maximum boxes per image: {MAX_BOXES}")
    print(f"Classes: {CLASS_MAPPING}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    model = build_model(args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_ROOT / "epoch_{epoch:02d}.keras"),
            save_weights_only=False,
            save_best_only=False,
        )
    ]

    if not args.skip_coco_metrics:
        callbacks.append(
            EvaluateCOCOMetricsCallback(
                validation_data=val_ds,
                save_path=SAVED_MODEL_PATH,
            )
        )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    if args.skip_coco_metrics:
        model.save(SAVED_MODEL_PATH)
        print(f"Saved model to {SAVED_MODEL_PATH}")


if __name__ == "__main__":
    main()
