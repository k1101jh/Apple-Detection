import os
import cv2
import numpy as np

if __name__ == "__main__":
    sample_image_path = r"/data/kjh/Apple Dataset/mixed_data/train/images/210817-Cam1-T02-003.JPG"

    original_image = cv2.imread(sample_image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = np.float32(image)

    H, S, V = cv2.split(image)
    hue_scale = 0.015 + 1
    saturation_scale = 0.7 + 1
    value_scale = 0.4 + 1
    H = (H * hue_scale) % 180
    S = np.clip(S * saturation_scale, 0, 255)
    V = np.clip(V * value_scale, 0, 255)

    image = cv2.merge([H, S, V])
    image = np.uint8(image)

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.hconcat([original_image, image])

    cv2.imwrite("image.png", image)
