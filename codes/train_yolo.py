import os
import hydra
import ultralytics
import torch
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ultralytics.checks()


def train():
    model = YOLO("yolov8m.pt")
    model.to("cuda")

    model.train(
        data="configs/data.yaml",
        name="yolov8m",
        epochs=1000,
        patience=200,
        batch=6,
        imgsz=800,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.2,
        scale=0.2,
        # perspective=0.1,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == "__main__":
    train()
