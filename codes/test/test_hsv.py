import os
import cv2
import copy
import numpy as np


def HSV_convert(image, hue_ratio, saturation_ratio, value_ratio):
    H, S, V = cv2.split(image)
    hue_m = 1 - hue_ratio
    hue_p = 1 + hue_ratio
    saturation_m = 1 - saturation_ratio
    saturation_p = 1 + saturation_ratio
    value_m = 1 - value_ratio
    value_p = 1 + value_ratio

    # minus_image
    image_m = cv2.merge([(H * hue_m) % 180, np.clip(S * saturation_m, 0, 255), np.clip(V * value_m, 0, 255)])
    image_m = np.uint8(image_m)
    image_m = cv2.cvtColor(image_m, cv2.COLOR_HSV2BGR)

    # plus_image
    image_p = cv2.merge([(H * hue_p) % 180, np.clip(S * saturation_p, 0, 255), np.clip(V * value_p, 0, 255)])
    image_p = np.uint8(image_p)
    image_p = cv2.cvtColor(image_p, cv2.COLOR_HSV2BGR)

    return image_m, image_p


if __name__ == "__main__":
    sample_image_path = (
        r"/data/kjh/Apple Dataset/mixed_data(RDA, WSU2019_w_exclude)/train/images/210817-Cam1-T02-003.JPG"
    )
    hue_ratio = 0
    saturation_ratio = 0.7
    value_ratio = 0.7

    original_image = cv2.imread(sample_image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = np.float32(image)

    image1, image2 = HSV_convert(image, hue_ratio, saturation_ratio, value_ratio)
    image = cv2.hconcat([original_image, image1, image2])

    cv2.imwrite("image.png", image)
