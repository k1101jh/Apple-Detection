import os
import glob
import pandas
import json
from PIL import Image
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class KFujiRGBDSDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform):
        self.dataset_type = dataset_type
        data_path_cfg = dataset_cfg[dataset_type]

        filelist = self.read_txt(data_path_cfg.filelist_txt_path)

        self.img_filelist = [os.path.join(dataset_cfg.dataset_path, f"{filename}hr.jpg") for filename in filelist]
        self.img_filelist.sort()

        self.csv_filelist = [os.path.join(dataset_cfg.annotation_path, f"{filename}.csv") for filename in filelist]
        self.csv_filelist.sort()

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = []

        for csv_file in self.csv_filelist:
            self.img_bboxes.append(self.read_csv(csv_file))

    def read_txt(self, txt_filepath):
        with open(txt_filepath, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]
        return lines

    def read_csv(self, csv_filepath):
        bboxes = []
        with open(csv_filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                bbox = line.split(",")[1:-1]
                bbox = [float(pos) for pos in bbox]
                # [x, y, w, h] to [min_x, max_y, max_x, min_y](top-left, bottom-right)
                bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]
                bbox = [round(pos) for pos in bbox]
                bboxes.append(bbox)

        return bboxes

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
