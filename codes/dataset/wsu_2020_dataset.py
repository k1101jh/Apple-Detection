import os
import glob
import xml
import numpy as np
from send2trash import send2trash
from PIL import Image
from xml.etree import ElementTree
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WSU2020Dataset(Dataset):
    def __init__(self, dataset_cfg, dataset_type, transform):
        self.dataset_type = dataset_type
        if self.dataset_type == "train":
            data_path_cfg = dataset_cfg.train
        else:
            raise Exception(f"해당 데이터셋에는 {dataset_type} 타입의 데이터가 없습니다.")

        img_pathname = os.path.join(data_path_cfg.dataset_path, "*.png")
        self.img_filelist = glob.glob(img_pathname)
        self.img_filelist.sort()

        xml_pathname = os.path.join(data_path_cfg.gt_path, "*.xml")
        self.xml_filelist = glob.glob(xml_pathname)
        self.xml_filelist.sort()

        self.num_images = len(self.img_filelist)
        self.image_transform = transform

        self.img_bboxes = []

        for csv_file in self.xml_filelist:
            self.img_bboxes.append(self.read_xml(csv_file))

    def read_xml(self, xml_filepath):
        tree = ElementTree.parse(xml_filepath)

        object_elements = tree.findall("object")

        bboxes = []

        for object_element in object_elements:
            bbox_element = object_element.find("bndbox")
            min_x = int(bbox_element.find("xmin").text)
            min_y = int(bbox_element.find("ymin").text)
            max_x = int(bbox_element.find("xmax").text)
            max_y = int(bbox_element.find("ymax").text)
            bboxes.append([min_x, max_y, max_x, min_y])

        return bboxes

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = Image.open(self.img_filelist[idx])
        bboxes = self.img_bboxes[idx]

        image = self.image_transform(image)

        return [image, bboxes, self.img_filelist[idx]]
