import os
import hydra
import ultralytics
import torch
import glob
from PIL import Image
from PIL import ImageOps
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
ultralytics.checks()

if __name__ == "__main__":
    model_path = "runs/detect/yolov8m_mixed_data_800/weights/best.pt"
    model = YOLO(model_path)
    model.to("cuda")

    os.makedirs("datasets/results", exist_ok=True)
    img_list = glob.glob("datasets/*.jpg")

    for img_path in img_list:
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        results = model.predict(image)

        for r in results:
            im_array = r.plot(labels=False)
            im = Image.fromarray(im_array[..., ::-1])
            im.save(os.path.join("datasets/results", os.path.basename(img_path)))
