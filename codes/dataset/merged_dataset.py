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


class MergedDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform, visibilities=["good"]):
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            data_path_cfg = dataset_cfg.train
        elif self.dataset_type == "test":
            data_path_cfg = dataset_cfg.test
        else:
            raise Exception(f"해당 데이터셋에는 {dataset_type} 타입의 데이터가 없습니다.")

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        txt_pathname = os.path.join(data_path_cfg.gt_path, "*")
        self.txt_filelist = glob.glob(txt_pathname)
        self.txt_filelist.sort()

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = []
        for txt_file in self.txt_filelist:
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # class id, cx, cy, w, h

            bboxes = []
            for line in lines[:-1]:
                _, cx, cy, w, h = line.split(" ")
                bboxes.append([float(cx), float(cy), float(w), float(h)])
            self.img_bboxes.append(bboxes)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        image = ImageOps.exif_transpose(image)
        bboxes = self.img_bboxes[idx]

        width, height = image.size
        bboxes = np.array(bboxes)

        if len(bboxes):
            bboxes[:, 0] *= width
            bboxes[:, 1] *= height
            bboxes[:, 2] *= width
            bboxes[:, 3] *= height

            bboxes[:, 0], bboxes[:, 2] = bboxes[:, 0] - (bboxes[:, 2] / 2), bboxes[:, 0] + (bboxes[:, 2] / 2)
            bboxes[:, 1], bboxes[:, 3] = bboxes[:, 1] - (bboxes[:, 3] / 2), bboxes[:, 1] + (bboxes[:, 3] / 2)

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
