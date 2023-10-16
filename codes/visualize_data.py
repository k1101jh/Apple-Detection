import os
import hydra
import glob
import argparse
import cv2
import csv
import json
import pandas
import matplotlib.pyplot as plt
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("merge", lambda x, y: x + y)

bbox_color = (124, 252, 0)
text_color = (15, 15, 240)
imshow_height = 800


def visualize_Fuji_SfM():
    initialize(config_path="../configs", job_name="visualize_Fuji_SfM")
    cfg = compose(config_name="config", overrides=["dataset=Fuji-SfM"])
    print(OmegaConf.to_yaml(cfg.dataset))

    def read_csv(csv_filepath):
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

        # with open(csv_filepath, 'r', encoding='utf-8') as f:
        #     reader = csv.reader(f)

    img_pathname = os.path.join(cfg.dataset.training_dataset_path, "*.jpg")
    img_filelist = glob.glob(img_pathname)
    img_filelist.sort()

    csv_pathname = os.path.join(cfg.dataset.training_dataset_path, "*.csv")
    csv_filelist = glob.glob(csv_pathname)
    csv_filelist.sort()

    num_images = len(img_filelist)
    cur_idx = 0

    while True:
        image = cv2.imread(img_filelist[cur_idx])
        bboxes = read_csv(csv_filelist[cur_idx])

        # cv2.putText(image, , (10, 10), 2, 2, (15, 15, 240), 2, )
        for bbox in bboxes:
            cv2.rectangle(image, [round(bbox[0]), round(bbox[1])], [round(bbox[2]), round(bbox[3])], bbox_color, 2)

        height, width = image.shape[:2]
        resize_ratio = imshow_height / height
        resized_width = round(width * resize_ratio)
        image = cv2.resize(image, dsize=(resized_width, imshow_height), interpolation=cv2.INTER_CUBIC)

        cv2.putText(
            image,
            f"{cur_idx + 1} / {num_images}  {os.path.basename(img_filelist[cur_idx])}",
            (10, 50),
            2,
            1,
            text_color,
            2,
            cv2.LINE_8,
        )

        cv2.imshow(f"image_samples", image)

        key_inp = cv2.waitKey(0) & 0xFF
        if key_inp == ord("a"):
            cur_idx = cur_idx - 1 if cur_idx > 0 else num_images - 1
        elif key_inp == ord("d"):
            cur_idx = cur_idx + 1 if cur_idx < num_images - 1 else 0
        elif key_inp == ord("q"):
            break


def visualize_MinneApple():
    initialize(config_path="../configs", job_name="visualize_MinneApple")
    cfg = compose(config_name="config", overrides=["dataset=MinneApple"])
    print(OmegaConf.to_yaml(cfg.dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Data")
    parser.add_argument("--data", help="Dataset name to visualize", type=str, default="Fuji-SfM")
    args = parser.parse_args()
    if args.data == "Fuji-SfM":
        visualize_Fuji_SfM()
    elif args.data == "MinneApple":
        visualize_MinneApple()
