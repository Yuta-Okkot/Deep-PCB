from io import BytesIO
from pathlib import Path

import math
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final" / "pcb_fault_detector_best.pt"
YOLO_RUNS_ROOT = PROJECT_ROOT / "models" / "yolo_runs"
YOLO_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "yolo"

APP_USER = "admin"
APP_PASSWORD = "pcb123"

CLASS_NAMES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole",
}


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f6f8fb;
        }
        div[data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.96);
        }
        .block-container {
            padding-top: 4.2rem;
            max-width: 1220px;
        }
        .hero {
            background: linear-gradient(135deg, #0f766e 0%, #164e63 55%, #1f2937 100%);
            color: white;
            padding: 26px 30px;
            border-radius: 8px;
            margin-bottom: 18px;
            border: 1px solid rgba(255, 255, 255, 0.22);
        }
        .hero h1 {
            margin: 0 0 8px 0;
            color: white;
            font-size: 2.1rem;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: #dff7f4;
            font-size: 1rem;
        }
        .panel {
            background: #ffffff;
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #d9e2ec;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }
        .status-strip {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            padding: 10px 12px;
            border-radius: 8px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .note {
            background: #f8fafc;
            border-left: 4px solid #0f766e;
            padding: 12px 14px;
            border-radius: 6px;
            color: #1f2937;
        }
        .nav-wrap {
            padding: 12px 0 26px 0;
            margin: 0;
        }
        .gallery-card {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 12px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .gallery-title {
            font-size: 0.82rem;
            color: #334155;
            font-weight: 700;
            margin: 6px 0 2px 0;
            word-break: break-word;
        }
        .gallery-meta {
            color: #64748b;
            font-size: 0.78rem;
            margin-bottom: 6px;
        }
        .explain-box {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .explain-box h3 {
            margin-top: 0;
            color: #0f766e;
        }
        .upload-panel {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 18px;
        }
        .upload-heading {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .upload-copy {
            color: #64748b;
            margin-bottom: 12px;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid #0f766e;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_navigation():
    labels = ["Scanner", "Dataset", "Model Explanation"]
    if "page" not in st.session_state:
        st.session_state.page = labels[0]

    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    columns = st.columns([1, 1, 1, 0.8])
    for column, label in zip(columns[:3], labels):
        button_type = "primary" if st.session_state.page == label else "secondary"
        if column.button(label, type=button_type, use_container_width=True):
            st.session_state.page = label
            st.rerun()

    if columns[3].button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.page = labels[0]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def find_latest_best_model() -> Path | None:
    if FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH

    candidates = sorted(
        YOLO_RUNS_ROOT.glob("*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_latest_run_dir() -> Path | None:
    candidates = sorted(
        [path for path in YOLO_RUNS_ROOT.glob("*") if path.is_dir() and (path / "results.csv").exists()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@st.cache_resource
def load_model(model_path: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics")
    return YOLO(model_path)


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def results_to_dataframe(result) -> pd.DataFrame:
    rows = []

    if result.boxes is None or len(result.boxes) == 0:
        return pd.DataFrame(columns=["defect", "confidence", "x1", "y1", "x2", "y2"])

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for class_id, confidence, box in zip(classes, confidences, boxes):
        x1, y1, x2, y2 = box
        rows.append(
            {
                "defect": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": round(float(confidence), 3),
                "x1": round(float(x1), 1),
                "y1": round(float(y1), 1),
                "x2": round(float(x2), 1),
                "y2": round(float(y2), 1),
            }
        )

    return pd.DataFrame(rows)


def run_detection(model, image: Image.Image, confidence: float, iou: float, image_size: int, device: str):
    return model.predict(
        source=image,
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        verbose=False,
    )[0]


def require_login():
    if st.session_state.get("authenticated"):
        return

    hero(
        "PCB Fault Detection",
        "Secure access to the AI inspection dashboard for printed circuit board defects.",
    )

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == APP_USER and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.markdown('<div class="note">Default demo login: admin / pcb123</div>', unsafe_allow_html=True)
    st.stop()


def scanner_page():
    hero(
        "PCB Defect Scanner",
        "Upload a PCB image and identify open, short, mousebite, spur, copper, and pin-hole defects.",
    )

    model_path = find_latest_best_model()

    with st.sidebar:
        st.header("Inspection Controls")

        if model_path is None:
            st.error("No trained model found.")
            st.caption("Expected a best.pt file in models/final or models/yolo_runs.")
            st.stop()

        selected_model_path = str(model_path)
        st.markdown('<div class="status-strip">Model loaded</div>', unsafe_allow_html=True)
        confidence = st.slider("Detection sensitivity", 0.05, 0.90, 0.25, 0.05)
        iou = st.slider("Box overlap filtering", 0.10, 0.90, 0.70, 0.05)
        image_size = st.selectbox("Inspection resolution", [416, 512, 640], index=0)
        device_choice = st.selectbox("Processing mode", ["GPU acceleration", "CPU fallback"], index=0)
        device = "0" if device_choice == "GPU acceleration" else "cpu"
        st.caption("Lower confidence finds more possible defects. Higher confidence shows fewer, stronger detections.")

    st.markdown(
        """
        <div class="upload-panel">
            <div class="upload-heading">Start A New PCB Inspection</div>
            <div class="upload-copy">
            Upload a PCB test image. The detector will mark visible faults and summarize the detected defect types.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "PCB inspection image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        preview_left, preview_right = st.columns([1.2, 1])
        with preview_left:
            st.markdown(
                """
                <div class="note">
                Waiting for an image. Use a clear, top-down PCB crop similar to the DeepPCB test images for best results.
                </div>
                """,
                unsafe_allow_html=True,
            )
        with preview_right:
            st.markdown(
                """
                <div class="panel">
                <b>Inspection output</b><br>
                After upload, this page will show the original PCB, the annotated result, defect counts, confidence scores,
                and coordinates for every detected fault.
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    image = Image.open(uploaded_file).convert("RGB")

    try:
        model = load_model(selected_model_path)
    except Exception as error:
        st.error(f"Could not load model: {error}")
        return

    left, right = st.columns(2)

    with left:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    with st.spinner("Scanning PCB..."):
        result = run_detection(model, image, confidence, iou, image_size, device)
        annotated_array = result.plot()
        annotated_image = Image.fromarray(annotated_array[..., ::-1])
        detections = results_to_dataframe(result)

    with right:
        st.subheader("Detected Defects")
        st.image(annotated_image, use_container_width=True)

    st.divider()

    total_detections = len(detections)
    unique_defects = detections["defect"].nunique() if total_detections else 0

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Defects", total_detections)
    metric_2.metric("Defect Types", unique_defects)
    metric_3.metric("Confidence", f"{confidence:.2f}")

    if detections.empty:
        st.success("No defects detected at the current confidence threshold.")
    else:
        st.subheader("Inspection Results")
        st.dataframe(detections, use_container_width=True, hide_index=True)

        summary = detections.groupby("defect").size().reset_index(name="count")
        st.subheader("Defect Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.download_button(
            label="Download annotated image",
            data=image_to_png_bytes(annotated_image),
            file_name="pcb_detection_result.png",
            mime="image/png",
        )


def evaluation_metrics(run_dir: Path):
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return None

    data = pd.read_csv(results_path)
    data.columns = [column.strip() for column in data.columns]
    if data.empty:
        return None

    last = data.iloc[-1]
    metric_columns = {
        "Precision": "metrics/precision(B)",
        "Recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }

    metrics = {}
    for label, column in metric_columns.items():
        if column in last:
            metrics[label] = float(last[column])
    return metrics


def parse_yolo_label_file(label_path: Path) -> pd.DataFrame:
    rows = []
    if not label_path.exists():
        return pd.DataFrame(columns=["defect", "class_id", "x_center", "y_center", "width", "height"])

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = map(float, parts[1:])
        rows.append(
            {
                "defect": CLASS_NAMES.get(class_id, str(class_id)),
                "class_id": class_id,
                "x_center": round(x_center, 4),
                "y_center": round(y_center, 4),
                "width": round(width, 4),
                "height": round(height, 4),
            }
        )
    return pd.DataFrame(rows)


def dataset_stats():
    rows = []
    for split_name in ["train", "val", "test"]:
        image_dir = YOLO_DATASET_ROOT / "images" / split_name
        label_dir = YOLO_DATASET_ROOT / "labels" / split_name
        image_count = len(list(image_dir.glob("*"))) if image_dir.exists() else 0
        label_count = len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0
        defect_count = 0
        if label_dir.exists():
            for label_path in label_dir.glob("*.txt"):
                defect_count += sum(1 for line in label_path.read_text().splitlines() if line.strip())
        rows.append({"split": split_name, "images": image_count, "label files": label_count, "defect boxes": defect_count})
    return pd.DataFrame(rows)


def defect_count_for_image(label_dir: Path, image_path: Path) -> int:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def draw_label_overlay(image_path: Path, label_path: Path) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#a855f7", "#ec4899"]

    labels = parse_yolo_label_file(label_path)
    for _, row in labels.iterrows():
        box_width = row["width"] * width
        box_height = row["height"] * height
        x_center = row["x_center"] * width
        y_center = row["y_center"] * height
        x1 = x_center - box_width / 2
        y1 = y_center - box_height / 2
        x2 = x_center + box_width / 2
        y2 = y_center + box_height / 2
        color = colors[int(row["class_id"]) % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), row["defect"], fill=color)

    return image


def dataset_page():
    hero(
        "Training Dataset Browser",
        "Inspect the images and YOLO labels used to train and validate the PCB defect detector.",
    )

    stats = dataset_stats()
    st.subheader("Dataset Summary")
    st.dataframe(stats, use_container_width=True, hide_index=True)

    control_1, control_2, control_3 = st.columns([1, 1, 1])
    split_name = control_1.selectbox("Dataset split", ["train", "val", "test"], index=0)
    gallery_size = control_2.selectbox("Gallery size", [6, 9, 12, 18, 24], index=1)
    start_index = control_3.number_input("Start index", min_value=0, value=0, step=gallery_size)

    image_dir = YOLO_DATASET_ROOT / "images" / split_name
    label_dir = YOLO_DATASET_ROOT / "labels" / split_name

    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg")))
    if not images:
        st.warning(f"No images found in {image_dir}")
        return

    start_index = min(int(start_index), max(0, len(images) - 1))
    gallery_images = images[start_index : start_index + gallery_size]

    st.subheader("PCB Gallery")
    st.caption(f"Showing {len(gallery_images)} samples from {split_name}, starting at index {start_index}.")

    selected_name = st.session_state.get("selected_dataset_image", gallery_images[0].name)
    columns_per_row = 3
    for row_start in range(0, len(gallery_images), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, image_path in zip(columns, gallery_images[row_start : row_start + columns_per_row]):
            label_path = label_dir / f"{image_path.stem}.txt"
            count = defect_count_for_image(label_dir, image_path)
            with column:
                st.markdown('<div class="gallery-card">', unsafe_allow_html=True)
                st.image(str(image_path), use_container_width=True)
                st.markdown(f'<div class="gallery-title">{image_path.name}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="gallery-meta">{count} labeled defects</div>', unsafe_allow_html=True)
                if st.button("Inspect", key=f"inspect_{split_name}_{image_path.name}", use_container_width=True):
                    st.session_state.selected_dataset_image = image_path.name
                    selected_name = image_path.name
                st.markdown("</div>", unsafe_allow_html=True)

    selected_image = image_dir / selected_name
    if not selected_image.exists():
        selected_image = gallery_images[0]
        selected_name = selected_image.name
    selected_label = label_dir / f"{selected_image.stem}.txt"

    st.divider()
    st.subheader("Selected Sample")
    left, right = st.columns([1.1, 1])
    with left:
        overlay = st.checkbox("Show label overlay", value=True)
        if overlay:
            st.image(draw_label_overlay(selected_image, selected_label), use_container_width=True)
        else:
            st.image(str(selected_image), use_container_width=True)

    labels = parse_yolo_label_file(selected_label)
    with right:
        st.subheader("YOLO Labels")
        st.caption(str(selected_label))
        if labels.empty:
            st.info("No labels found for this image.")
        else:
            st.dataframe(labels, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note">
        YOLO labels store normalized box coordinates: class, x-center, y-center, width, and height.
        Values are between 0 and 1, so they remain valid if the training image is resized.
        </div>
        """,
        unsafe_allow_html=True,
    )


def explanation_page():
    hero(
        "Model Goal And Evaluation",
        "Understand what the detector learns and how to read the training graphs.",
    )

    st.markdown(
        """
        <div class="explain-box">
        <h3>What The Model Does</h3>
        <p>
        The dashboard uses an object detection model trained on the DeepPCB dataset.
        Its task is to inspect a PCB image and locate manufacturing defects with
        bounding boxes. Each detection includes a defect type and a confidence score.
        This is more useful than simple classification because the operator can see
        where the fault appears on the board, not only whether the board is defective.
        </p>
        <p>
        The six learned defect classes are open, short, mousebite, spur, copper, and
        pin-hole. In an industrial inspection workflow, this helps support fast
        quality-control decisions and reduces the chance of missing small visual defects.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="explain-box">
        <h3>How To Read The Metrics</h3>
        <p><b>Precision</b> measures how trustworthy the detections are. High precision means
        that when the model draws a box, it is usually detecting a real defect.</p>
        <p><b>Recall</b> measures how many real defects are found. For PCB inspection, recall
        is especially important because missing a defect can allow a faulty board to pass.</p>
        <p><b>mAP50</b> is the average detection quality when a predicted box overlaps the real
        box by at least 50 percent. It is a good general indicator of whether the model finds defects.</p>
        <p><b>mAP50-95</b> is stricter. It averages performance across several overlap thresholds,
        so it rewards boxes that are placed more accurately around the defect region.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    run_dir = find_latest_run_dir()
    if run_dir is None:
        st.warning("No YOLO training run with results.csv was found.")
        return

    st.caption(f"Evaluation run: {run_dir}")

    metrics = evaluation_metrics(run_dir)
    if metrics:
        columns = st.columns(len(metrics))
        for column, (label, value) in zip(columns, metrics.items()):
            column.metric(label, f"{value:.3f}")

    st.subheader("Training Curves")
    results_image = run_dir / "results.png"
    if results_image.exists():
        st.image(str(results_image), use_container_width=True)
        st.write(
            "The loss curves show how the model improves during training. Box loss "
            "measures localization error, class loss measures defect-type error, and "
            "DFL loss helps refine bounding-box quality. The metric curves show whether "
            "validation performance improves on unseen images. When the validation score "
            "stops improving, early stopping keeps the best checkpoint as best.pt."
        )

    graph_info = [
        (
            "Precision-Recall Curve",
            run_dir / "BoxPR_curve.png",
            "A curve closer to the top-right means the detector keeps high precision while finding more defects.",
        ),
        (
            "F1 Confidence Curve",
            run_dir / "BoxF1_curve.png",
            "This helps choose a confidence threshold that balances false alarms and missed defects.",
        ),
        (
            "Confusion Matrix",
            run_dir / "confusion_matrix_normalized.png",
            "This shows which defect classes are correctly identified and which classes are confused.",
        ),
    ]

    for title, image_path, explanation in graph_info:
        if image_path.exists():
            st.subheader(title)
            st.image(str(image_path), use_container_width=True)
            st.write(explanation)

    st.subheader("Engineering Interpretation")
    st.markdown(
        """
        <div class="explain-box">
        <h3>Engineering Interpretation</h3>
        <p>
        In process engineering and quality control, the model should be judged by how it
        supports inspection decisions. A high recall reduces the probability of releasing
        faulty PCBs. A high precision reduces unnecessary manual re-checks. A strong mAP50-95
        indicates that the bounding boxes are tight enough to help operators locate the exact
        defect region.
        </p>
        <p>
        If the model has high mAP50 but lower mAP50-95, it usually finds the correct defect
        but the box may not be perfectly tight. If recall is lower for a specific class, that
        class should receive more training samples, higher-resolution training, or targeted
        augmentation. If the confusion matrix shows two classes being mixed, those defects
        likely have similar visual patterns and may need clearer annotations or more examples.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="PCB Fault Detection", layout="wide")
apply_theme()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

require_login()

top_navigation()

if st.session_state.page == "Scanner":
    scanner_page()
elif st.session_state.page == "Dataset":
    dataset_page()
else:
    explanation_page()
