import os
import glob
import pandas
import json
import csv
import natsort
import numpy as np
from send2trash import send2trash
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WSU2019Dataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform):
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            data_path_cfg = dataset_cfg.train
        else:
            raise Exception(f"해당 데이터셋에는 {dataset_type} 타입의 데이터가 없습니다.")

        # 데이터셋 오류 확인 및 해결
        if os.path.exists(os.path.join(data_path_cfg.dataset_path, "image-33(1).png")):
            print(os.path.join(data_path_cfg.dataset_path, "image-33(1).png"))
            send2trash(os.path.join(data_path_cfg.dataset_path, "image-33(1).png").replace("/", "\\"))

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.png")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist = natsort.natsorted(self.img_filelist)

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = self.read_csv(data_path_cfg.gt_path)

    def read_csv(self, csv_filepath):
        all_image_bboxes = []
        with open(csv_filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                bboxes = []
                splitted = line.split(", ")[1:-1]
                bbox = []
                for i, pos_str in enumerate(splitted):
                    bbox.append(int(pos_str))
                    if (i + 1) % 4 == 0:
                        # [min_x, max_y, max_x, min_y]
                        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]
                        bboxes.append(bbox[:])
                        bbox = []
                all_image_bboxes.append(bboxes)
        return all_image_bboxes

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
