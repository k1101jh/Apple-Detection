import os
import hydra
import ultralytics
import torch
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ultralytics.checks()


def train():
    model_path = "runs/detect/yolov8m_GFB_WSU2019_KFuji_800/weights/best.pt"
    model = YOLO(model_path)
    model.to("cuda")
    metrics = model.val()
    print(metrics.box.maps)


if __name__ == "__main__":
    train()
