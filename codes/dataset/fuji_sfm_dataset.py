import os
import glob
import pandas
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FujiSfMDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform):
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

        csv_pathname = os.path.join(data_path_cfg.dataset_path, "*.csv")
        self.csv_filelist = glob.glob(csv_pathname)
        self.csv_filelist.sort()

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
