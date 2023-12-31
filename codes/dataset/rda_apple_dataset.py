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
    def __init__(self, dataset_cfg, dataset_type, transform, visibilities=["good"]):
        self.dataset_type = dataset_type
        data_path_cfg = dataset_cfg[dataset_type]

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

            visible_bboxes = []
            for visibility in visibilities:
                visible_bboxes += json_object[visibility]
            self.img_bboxes.append(visible_bboxes)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        image = ImageOps.exif_transpose(image)
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
