import os
import hydra
import ultralytics
import torch
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
ultralytics.checks()


def train():
    model = YOLO("yolov8m.pt")
    model.to("cuda")

    imgsz = 1200

    model.train(
        data="configs/mixed_data(RDA, WSU2019, MinneApple, KFuji, Fuji).yaml",
        name=f"yolov8m_mixed_data_{imgsz}",
        epochs=1000,
        patience=200,
        batch=5,
        imgsz=imgsz,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.7,
        degrees=15,
        translate=0.3,
        scale=0.3,
        # perspective=0.1,
        fliplr=0.5,
        mosaic=0.7,
    )


if __name__ == "__main__":
    train()
