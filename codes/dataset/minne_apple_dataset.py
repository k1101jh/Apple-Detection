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
    def __init__(self, dataset_cfg, dataset_type, transform, exclude_bad_images=False):
        self.dataset_type = dataset_type
        data_path_cfg = dataset_cfg[dataset_type]

        if not os.path.exists(data_path_cfg.gt_path):
            self.generate_gt(data_path_cfg.mask_path, data_path_cfg.gt_path)

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.png")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        img_bboxes = {}
        for i in range(len(self.img_filelist)):
            img_bboxes[i] = []

        self.files_to_exclude = []
        if exclude_bad_images:
            self.files_to_exclude = data_path_cfg.files_to_exclude
            self.img_filelist = [
                img_file
                for img_file in self.img_filelist
                if os.path.splitext(os.path.basename(img_file))[0] not in self.files_to_exclude
            ]

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        with open(data_path_cfg.gt_path, "r", encoding="utf-8") as f:
            json_object = json.load(f)

        image_ids = []
        for image_info in json_object["images"]:
            if os.path.splitext(image_info["filename"])[0] in self.files_to_exclude:
                image_ids.append(image_info["id"])

        for bbox_dict in json_object["annotations"]:
            bbox = bbox_dict["bbox"]
            bbox[2] += bbox[0]
            bbox[1], bbox[3] = bbox[1] + bbox[3], bbox[1]
            img_bboxes[bbox_dict["image_id"]].append(bbox)

        ids_to_del = []
        for key, val in img_bboxes.items():
            if key in image_ids:
                ids_to_del.append(key)

        for id_to_del in ids_to_del:
            del img_bboxes[id_to_del]

        assert len(self.img_filelist) == len(
            img_bboxes
        ), f"img 개수와 gt 개수가 맞지 않습니다. img: {len(self.img_filelist)}개, gt: {len(self.img_bboxes)}개"

        self.img_bboxes = list(img_bboxes.values())

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
