from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET_ROOT = PROJECT_ROOT.parent / "DeepPCB-master" / "PCBData"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
VISUALIZATION_ROOT = OUTPUTS_ROOT / "visualizations"

CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "copper",
    6: "pin-hole",
}

