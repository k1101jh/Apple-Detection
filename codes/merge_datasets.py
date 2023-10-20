import os
import glob
import shutil
from tqdm import tqdm
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from torchvision import transforms

from dataset.minne_apple_dataset import MinneAppleDataset
from dataset.wsu_2019_dataset import WSU2019Dataset
from dataset.wsu_2020_dataset import WSU2020Dataset
from dataset.fuji_sfm_dataset import FujiSfMDataset
from dataset.kfuji_rgb_ds_dataset import KFujiRGBDSDataset
from dataset.rda_apple_dataset import RDAAppleDataset


def merge_datasets(dest_path, datasets, dataset_type):
    def xyxy_to_cxywh(xyxy, width, height):
        cx = ((xyxy[0] + xyxy[2]) / 2) / width
        cy = ((xyxy[1] + xyxy[3]) / 2) / height
        w = (xyxy[2] - xyxy[0]) / width
        h = (xyxy[1] - xyxy[3]) / height
        return [cx, cy, w, h]

    def save_txt(filepath, bboxes, width, height):
        # yolov8 txt format: [class_id center_x center_y width height]
        class_id = 0
        with open(filepath, "w") as f:
            for bbox in bboxes:
                cxywh = xyxy_to_cxywh(bbox, width, height)
                f.write(f"{class_id} {cxywh[0]} {cxywh[1]} {cxywh[2]} {cxywh[3]}\n")

    image_path = os.path.join(dest_path, dataset_type, "images")
    label_path = os.path.join(dest_path, dataset_type, "labels")
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for dataset in tqdm(datasets, desc="dataset", position=0, leave=True):
        for image, bboxes, filepath in tqdm(dataset, desc="image", position=1, leave=False):
            image_dest_path = os.path.join(image_path, os.path.basename(filepath))
            label_dest_path = os.path.join(label_path, os.path.splitext(os.path.basename(filepath))[0] + ".txt")
            width, height = image.size
            shutil.copy(filepath, image_dest_path)

            save_txt(label_dest_path, bboxes, width, height)


def no_action_transform(x):
    return x


if __name__ == "__main__":
    dataset_path = "datasets/mixed_data"
    os.makedirs(dataset_path, exist_ok=True)

    initialize(config_path="../configs", job_name="merge_dataset")
    dataset_type = "train"

    datasets = []
    # dataset_cfg = compose(config_name="config", overrides=[f"dataset=MinneApple"]).dataset
    # datasets.append(MinneAppleDataset(dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform)))

    dataset_cfg = compose(config_name="config", overrides=[f"dataset=WSU2019"]).dataset
    datasets.append(
        WSU2019Dataset(
            dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform), exclude_bad_images=True
        )
    )

    # dataset_cfg = compose(config_name="config", overrides=[f"dataset=WSU2020"]).dataset
    # datasets.append(WSU2020Dataset(dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform)))

    # dataset_cfg = compose(config_name="config", overrides=[f"dataset=Fuji-SfM"]).dataset
    # datasets.append(FujiSfMDataset(dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform)))

    # dataset_cfg = compose(config_name="config", overrides=[f"dataset=KFuji RGB-DS"]).dataset
    # datasets.append(KFujiRGBDSDataset(dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform)))

    dataset_cfg = compose(config_name="config", overrides=[f"dataset=RDA-Apple"]).dataset
    datasets.append(
        RDAAppleDataset(
            dataset_cfg, dataset_type, transform=transforms.Lambda(no_action_transform), visibilities=["good", "fair"]
        )
    )

    merge_datasets(dataset_path, datasets, dataset_type)
