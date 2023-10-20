import os
import glob
import pandas
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageOps
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RDAAppleDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform, visibility=["good"]):
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            data_path_cfg = dataset_cfg.train
        elif self.dataset_type == "test":
            data_path_cfg = dataset_cfg.test
        else:
            raise Exception(f"해당 데이터셋에는 {dataset_type} 타입의 데이터가 없습니다.")

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.jpg")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        json_pathname = os.path.join(data_path_cfg.gt_path, "*.json")
        self.json_filelist = glob.glob(json_pathname)
        self.json_filelist.sort()

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = []
        for json_file in self.json_filelist:
            with open(json_file, "r", encoding="utf-8") as f:
                json_object = json.load(f)

            for v in visibility:
                self.img_bboxes.append(json_object[v])

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        image = ImageOps.exif_transpose(image)
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
