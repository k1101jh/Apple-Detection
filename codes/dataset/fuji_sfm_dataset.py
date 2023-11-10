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


class FujiSfMDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform, exclude_bad_images=False):
        self.dataset_type = dataset_type
        data_path_cfg = dataset_cfg[dataset_type]

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.jpg")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        csv_pathname = os.path.join(data_path_cfg.dataset_path, "*.csv")
        self.csv_filelist = glob.glob(csv_pathname)
        self.csv_filelist.sort()

        self.files_to_exclude = []
        if exclude_bad_images:
            self.files_to_exclude = data_path_cfg.files_to_exclude
            self.img_filelist = [
                img_file
                for img_file in self.img_filelist
                if os.path.splitext(os.path.basename(img_file))[0] not in self.files_to_exclude
            ]
            self.csv_filelist = [
                csv_file
                for csv_file in self.csv_filelist
                if os.path.splitext(os.path.basename(csv_file))[0][5:] not in self.files_to_exclude
            ]

        assert len(self.img_filelist) == len(self.csv_filelist), "img 개수와 csv 개수가 맞지 않습니다."
        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = []

        for csv_file in self.csv_filelist:
            self.img_bboxes.append(self.read_csv(csv_file))

    def read_csv(self, csv_filepath):
        data = pandas.read_csv(csv_filepath)
        data = [json.loads(element) for element in data["region_shape_attributes"].to_list()]
        bboxes = []

        for polygon_data in data:
            all_points_x = polygon_data["all_points_x"]
            all_points_y = polygon_data["all_points_y"]

            min_x = min(all_points_x)
            max_x = max(all_points_x)
            min_y = min(all_points_y)
            max_y = max(all_points_y)

            # top-left, bottom-right
            bboxes.append([min_x, max_y, max_x, min_y])

        return bboxes

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
