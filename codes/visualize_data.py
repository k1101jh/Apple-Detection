import os
import hydra
import glob
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from torchvision import transforms

from dataset.minne_apple_dataset import MinneAppleDataset
from dataset.wsu_2019_dataset import WSU2019Dataset
from dataset.wsu_2020_dataset import WSU2020Dataset
from dataset.fuji_sfm_dataset import FujiSfMDataset

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)

bbox_color = (124, 252, 0)
text_color = (15, 15, 240)
imshow_height = 800


def visualize_dataset(dataset):
    """
    데이터 시각화

    Args:
        cfg (_type_): _description_
        dataset (Dataset): _description_
    """
    num_images = len(dataset)
    cur_idx = 0

    while True:
        image, bboxes, filename = dataset[cur_idx]
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]
        resize_ratio = imshow_height / height
        resized_width = round(width * resize_ratio)
        image = cv2.resize(image, dsize=(resized_width, imshow_height), interpolation=cv2.INTER_CUBIC)

        for bbox in bboxes:
            # min_x, max_y, max_x, min_y
            resized_bbox = [
                round(bbox[0] * resize_ratio),
                round(bbox[1] * resize_ratio),
                round(bbox[2] * resize_ratio),
                round(bbox[3] * resize_ratio),
            ]
            cv2.rectangle(image, [resized_bbox[0], resized_bbox[1]], [resized_bbox[2], resized_bbox[3]], bbox_color, 2)

        cv2.putText(
            image,
            f"{cur_idx + 1} / {num_images}",
            (10, 30),
            2,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{os.path.basename(filename)}",
            (10, 60),
            2,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(f"image_samples", image)

        key_inp = cv2.waitKey(0) & 0xFF
        if key_inp == ord("a"):
            cur_idx = cur_idx - 1 if cur_idx > 0 else num_images - 1
        elif key_inp == ord("d"):
            cur_idx = cur_idx + 1 if cur_idx < num_images - 1 else 0
        elif key_inp == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Data")
    parser.add_argument("--data", help="Dataset name to visualize", type=str, default="MinneApple")
    parser.add_argument(
        "--dataset-type", help="Datset type. examples: train, test, validation...", type=str, default="train"
    )
    args = parser.parse_args()

    initialize(config_path="../configs", job_name="visualize_dataset")
    cfg = compose(config_name="config", overrides=[f"dataset={args.data}"])
    print(OmegaConf.to_yaml(cfg.dataset))

    if args.data == "MinneApple":
        dataset = MinneAppleDataset(cfg.dataset, args.dataset_type, transform=transforms.Lambda(lambda x: x))
    elif args.data == "WSU2019":
        dataset = WSU2019Dataset(cfg.dataset, args.dataset_type, transform=transforms.Lambda(lambda x: x))
    elif args.data == "WSU2020":
        dataset = WSU2020Dataset(cfg.dataset, args.dataset_type, transform=transforms.Lambda(lambda x: x))
    elif args.data == "Fuji-SfM":
        dataset = FujiSfMDataset(cfg.dataset, args.dataset_type, transform=transforms.Lambda(lambda x: x))

    visualize_dataset(dataset)
