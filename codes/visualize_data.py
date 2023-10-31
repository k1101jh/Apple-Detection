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
from dataset.kfuji_rgb_ds_dataset import KFujiRGBDSDataset
from dataset.rda_apple_dataset import RDAAppleDataset
from dataset.merged_dataset import MergedDataset

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


def no_action_transform(x):
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Data")
    parser.add_argument("--data", help="Dataset name to visualize", type=str, default="KFuji RGB-DS")
    parser.add_argument(
        "--dataset-type", help="Datset type. examples: train, test, validation...", type=str, default="validation"
    )
    args = parser.parse_args()

    initialize(config_path="../configs", job_name="visualize_dataset")
    dataset_cfg = compose(config_name="config", overrides=[f"dataset={args.data}"]).dataset
    print(OmegaConf.to_yaml(dataset_cfg))

    if args.data == "MinneApple":
        dataset = MinneAppleDataset(
            dataset_cfg, args.dataset_type, transform=transforms.Lambda(no_action_transform), exclude_bad_images=True
        )
    elif args.data == "WSU2019":
        dataset = WSU2019Dataset(
            dataset_cfg, args.dataset_type, transform=transforms.Lambda(no_action_transform), exclude_bad_images=True
        )
    elif args.data == "WSU2020":
        dataset = WSU2020Dataset(dataset_cfg, args.dataset_type, transform=transforms.Lambda(no_action_transform))
    elif args.data == "Fuji-SfM":
        dataset = FujiSfMDataset(
            dataset_cfg, args.dataset_type, transform=transforms.Lambda(no_action_transform), exclude_bad_images=True
        )
    elif args.data == "KFuji RGB-DS":
        dataset = KFujiRGBDSDataset(dataset_cfg, args.dataset_type, transform=transforms.Lambda(no_action_transform))
    elif args.data == "RDA-Apple":
        dataset = RDAAppleDataset(
            dataset_cfg,
            args.dataset_type,
            transform=transforms.Lambda(no_action_transform),
            visibilities=["good", "fair"],
        )
    elif args.data == "MergedData":
        dataset = MergedDataset(
            dataset_cfg,
            args.dataset_type,
            transform=transforms.Lambda(no_action_transform),
        )

    visualize_dataset(dataset)
