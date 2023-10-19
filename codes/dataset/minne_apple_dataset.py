import os
import glob
import pandas
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MinneAppleDataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform):
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            data_path_cfg = dataset_cfg.train
        elif self.dataset_type == "test":
            data_path_cfg = dataset_cfg.test
        else:
            raise Exception(f"해당 데이터셋에는 {dataset_type} 타입의 데이터가 없습니다.")

        if not os.path.exists(data_path_cfg.gt_path):
            self.generate_gt(data_path_cfg.mask_path, data_path_cfg.gt_path)

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.png")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        csv_pathname = os.path.join(data_path_cfg.gt_path, "*.csv")
        self.csv_filelist = glob.glob(csv_pathname)
        self.csv_filelist.sort()

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        with open(data_path_cfg.gt_path, "r", encoding="utf-8") as f:
            json_object = json.load(f)

        self.img_bboxes = {}
        for i in range(self.num_images):
            self.img_bboxes[i] = []
        for bbox_dict in json_object["annotations"]:
            bbox = bbox_dict["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            self.img_bboxes[bbox_dict["image_id"]].append(bbox)

    def generate_gt(self, mask_path, gt_path):
        gt_dict = {}
        gt_dict["images"] = []
        gt_dict["annotations"] = []

        mask_pathname = os.path.join(mask_path, "*.png")
        mask_filelist = glob.glob(mask_pathname)
        mask_filelist.sort()

        for image_id, mask_file in enumerate(tqdm(mask_filelist, desc="Save mask to json....", leave=True)):
            mask = Image.open(mask_file)

            width, height = mask.size

            img_dict = {}
            img_dict["width"] = width
            img_dict["height"] = height
            img_dict["id"] = image_id
            img_dict["filename"] = os.path.basename(mask_file)

            gt_dict["images"].append(img_dict)

            colors = mask.getcolors()
            colors_dict = {}

            for color in colors:
                # [min_x, max_y, max_x, min_y]
                colors_dict[color[1]] = [width, 0, 0, height]

            del colors_dict[0]

            # shape: (height, width)
            mask = np.array(mask)

            for i in range(height):
                for j in range(width):
                    if mask[i, j] != 0:
                        colors_dict[mask[i, j]][0] = min(colors_dict[mask[i, j]][0], j)
                        colors_dict[mask[i, j]][1] = max(colors_dict[mask[i, j]][1], i)
                        colors_dict[mask[i, j]][2] = max(colors_dict[mask[i, j]][2], j)
                        colors_dict[mask[i, j]][3] = min(colors_dict[mask[i, j]][3], i)

            for color, bbox in colors_dict.items():
                # [min_x, max_y, max_x, min_y] 를 [min_x, min_y, w, h] 형식으로 수정
                bbox[2] -= bbox[0]
                bbox[1], bbox[3] = bbox[3], bbox[1]
                bbox[3] -= bbox[1]

                bbox_dict = {}
                bbox_dict["bbox"] = bbox
                bbox_dict["id"] = color
                bbox_dict["image_id"] = image_id
                gt_dict["annotations"].append(bbox_dict)

        with open(gt_path, "w") as gtfile:
            json.dump(gt_dict, gtfile)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
