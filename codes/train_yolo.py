import os
import hydra
import ultralytics
import torch
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ultralytics.checks()


def train():
    model = YOLO("yolov8n.pt")
    model.to("cuda")

    model.train(data="configs/data.yaml", batch=8, name="yolov8n")


if __name__ == "__main__":
    train()
