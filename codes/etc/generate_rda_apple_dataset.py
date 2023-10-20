import os
import glob
import pandas
import json
import random
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

bbox_color = {
    "good": (124, 252, 0),
    "fair": (0, 124, 252),
    "bad": (0, 0, 252),
}
text_color = (15, 15, 240)


def visualize_dataset(image_path, annotation_filepath, save_path=None):
    bbox_list = list(read_csv(annotation_filepath).items())

    num_images = len(bbox_list)
    cur_idx = 0
    imshow_height = 1000

    while True:
        filename, visibilities = list(bbox_list[cur_idx])
        image = cv2.imread(os.path.join(image_path, filename), cv2.IMREAD_COLOR)

        height, width = image.shape[:2]
        imshow_height = height
        resize_ratio = imshow_height / height
        resize_ratio = 1
        resized_width = round(width * resize_ratio)
        image = cv2.resize(image, dsize=(resized_width, imshow_height), interpolation=cv2.INTER_CUBIC)

        for visibility, bboxes in list(visibilities.items()):
            # min_x, max_y, max_x, min_y
            for bbox in list(bboxes):
                resized_bbox = [
                    round(bbox[0] * resize_ratio),
                    round(bbox[1] * resize_ratio),
                    round(bbox[2] * resize_ratio),
                    round(bbox[3] * resize_ratio),
                ]
                cv2.rectangle(
                    image,
                    [resized_bbox[0], resized_bbox[1]],
                    [resized_bbox[2], resized_bbox[3]],
                    bbox_color[visibility],
                    3,
                )

        if save_path:
            cv2.imwrite(os.path.join(save_path, filename), image)

        cv2.putText(
            image,
            f"{cur_idx + 1} / {num_images}",
            (10, 30),
            2,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            f"{os.path.basename(filename)}",
            (10, 60),
            2,
            0.8,
            text_color,
            2,
            cv2.LINE_AA,
        )

        # cv2.imshow(f"image_samples", image)

        # key_inp = cv2.waitKey(0) & 0xFF
        # if key_inp == ord("a"):
        #     cur_idx = cur_idx - 1 if cur_idx > 0 else num_images - 1
        # elif key_inp == ord("d"):
        #     cur_idx = cur_idx + 1 if cur_idx < num_images - 1 else 0
        # elif key_inp == ord("q"):
        #     break

        cur_idx += 1
        if cur_idx == 199:
            break


def read_csv(csv_filepath):
    data = pandas.read_csv(csv_filepath)
    filenames = data["filename"].to_list()
    bbox_data = [json.loads(element) for element in data["region_shape_attributes"].to_list()]
    visibility_data = [json.loads(element)["visibility"] for element in data["region_attributes"].to_list()]
    bbox_dict = {}

    for filename, bbox, visibility in zip(filenames, bbox_data, visibility_data):
        if not filename in bbox_dict:
            bbox_dict[filename] = {}
            bbox_dict[filename]["good"] = []
            bbox_dict[filename]["fair"] = []
            bbox_dict[filename]["bad"] = []

        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]

        min_x = x
        max_x = x + w
        min_y = y
        max_y = y + h

        # top-left, bottom-right
        bbox_dict[filename][visibility].append([min_x, max_y, max_x, min_y])

    return bbox_dict


def generate_dataset(image_path, annotation_filepath, dest_path, test_ratio, random_seed):
    bbox_dict = read_csv(annotation_filepath)

    # split train test
    train_filenames, test_filenames, train_labels, test_labels = train_test_split(
        list(bbox_dict.keys()), list(bbox_dict.values()), test_size=test_ratio, random_state=random_seed
    )

    train_image_dir_path = os.path.join(dest_path, "train", "images")
    train_label_dir_path = os.path.join(dest_path, "train", "labels")
    test_image_dir_path = os.path.join(dest_path, "test", "images")
    test_label_dir_path = os.path.join(dest_path, "test", "labels")

    os.makedirs(train_image_dir_path, exist_ok=True)
    os.makedirs(train_label_dir_path, exist_ok=True)
    os.makedirs(test_image_dir_path, exist_ok=True)
    os.makedirs(test_label_dir_path, exist_ok=True)

    for train_filename, train_label in tqdm(zip(train_filenames, train_labels)):
        shutil.copy(os.path.join(image_path, train_filename), os.path.join(train_image_dir_path, train_filename))
        with open(os.path.join(train_label_dir_path, os.path.splitext(train_filename)[0] + ".json"), "w") as f:
            json.dump(train_label, f, ensure_ascii=False, indent=4)

    for test_filename, test_label in tqdm(zip(test_filenames, test_labels)):
        shutil.copy(os.path.join(image_path, test_filename), os.path.join(test_image_dir_path, test_filename))
        with open(os.path.join(test_label_dir_path, os.path.splitext(test_filename)[0] + ".json"), "w") as f:
            json.dump(test_label, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    image_path = r"D:\DeepLearning\Dataset\RDA apple data\Apple_annotation\images"
    annotation_filepath = r"D:\DeepLearning\Dataset\RDA apple data\Apple_annotation\Apple21.csv"
    dest_path = r"D:\DeepLearning\Dataset\Apple\RDA_Apple"

    test_ratio = 0.2
    random_seed = 2023

    # generate_dataset(image_path, annotation_filepath, dest_path, test_ratio, random_seed)
    visualize_dataset(
        image_path, annotation_filepath, r"D:\DeepLearning\Dataset\RDA apple data\Apple_annotation\visualization"
    )
