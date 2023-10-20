import hydra
import ultralytics
from ultralytics import YOLO
from tqdm import tqdm

ultralytics.checks()


def train():
    model = YOLO("yolov8n.pt")


if __name__ == "__main__":
    train()
