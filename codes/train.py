import hydra
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm

ultralytics.checks()


def train():
    model = YOLO("checkpoints/yolov8n.pt")

    model.train(
        data="datasets/yolo_data.yaml",
        epochs=5,
        imgsz=640,
        project="datasets/mixed_data",
        name="mixed_data",
        exist_ok=True,
    )


if __name__ == "__main__":
    train()
