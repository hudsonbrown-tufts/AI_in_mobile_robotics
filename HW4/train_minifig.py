# Fine-tunes a YOLOv8 detector to find the LEGO minifigure.
#
# Dataset: Minifig.v2i.yolov8/ (Roboflow YOLOv8 export, class "Minifig",
# 101 train / 29 valid / 14 test images; v2 adds photos of the minifig on the car).
# Training progress and the final weights land under
# HW4/runs/detect/minifig_v2/weights/best.pt

from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

HERE = Path(__file__).parent
DATASET_DIR = HERE / "Minifig.v2i.yolov8"
EPOCHS = 100  # small dataset benefits from more epochs than the usual 50
IMAGE_SIZE = 640
DEVICE = 0  # first CUDA GPU (RTX 3060); falls back to CPU below if unavailable
RUN_NAME = "minifig_v2"


def write_resolved_yaml():
    # Roboflow's data.yaml uses "../train/images"-style paths, which don't
    # resolve from the dataset folder. Write a copy with absolute paths.
    with open(DATASET_DIR / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str(DATASET_DIR.resolve())
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    cfg["test"] = "test/images"
    cfg.pop("roboflow", None)
    out = DATASET_DIR / "data_resolved.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def train(data_yaml, device):
    # Start from COCO-pretrained weights and fine-tune -- much faster than
    # training from scratch, and works well with a few dozen images.
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(HERE / "runs" / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        device=device,
    )
    return model


if __name__ == "__main__":  # guard required on Windows for dataloader workers
    data_yaml = write_resolved_yaml()

    device = DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Training on device={device!r}"
          + (f" ({torch.cuda.get_device_name(0)})" if device != "cpu" else ""))

    try:
        model = train(data_yaml, device)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"Training on device={device!r} failed ({e}). Retrying on CPU...")
        model = train(data_yaml, "cpu")

    # Score the final weights on the held-out test split.
    best = HERE / "runs" / "detect" / RUN_NAME / "weights" / "best.pt"
    YOLO(best).val(data=str(data_yaml), split="test", device=device,
                   project=str(HERE / "runs" / "detect"), name=f"{RUN_NAME}_test",
                   exist_ok=True)

    print(f"Done. Weights saved to {best}")
