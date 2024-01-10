import os
import numpy as np
import hydra
import ultralytics
import torch
import glob
from PIL import Image
from PIL import ImageOps
from ultralytics import YOLO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
ultralytics.checks()

if __name__ == "__main__":
    model_path = "runs/detect/yolov8m_mixed_data_800/weights/best.pt"
    model = YOLO(model_path)
    model.to("cuda")

    dataset_name = "SensitivityAnalysis"
    dataset_dir = os.path.join("/data/kjh/Apple Dataset", dataset_name)

    result_dir = os.path.join("results", dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    vid_dir_list = glob.glob(f"{dataset_dir}/*")

    for vid_dir in vid_dir_list:
        vid_name = os.path.basename(vid_dir)
        img_result_dir = os.path.join(result_dir, vid_name, "images")
        os.makedirs(img_result_dir, exist_ok=True)
        inference_result_path = os.path.join(result_dir, vid_name, "inference_result.txt")
        result_file = open(inference_result_path, "w", encoding="utf-8")

        img_list = glob.glob(f"{vid_dir}/img1/*")
        img_list.sort()
        for frame, img_path in enumerate(img_list):
            image = Image.open(img_path)
            image = ImageOps.exif_transpose(image)
            results = model.predict(image, conf=0.05)

            for r in results:
                # im_array = r.plot(labels=False)
                # im = Image.fromarray(im_array[..., ::-1])
                # im.save(os.path.join(img_result_dir, os.path.basename(img_path)))

                for box, conf in zip(r.boxes.xywh, r.boxes.conf):
                    box_xywh = torch.tensor([box[0] - box[2] / 2, box[1] - box[3] / 2, box[2], box[3]])

                    result_file.write(
                        f"{frame + 1},{0},{box_xywh[0]:.2f},{box_xywh[1]:.2f},{box_xywh[2]:.2f},{box_xywh[3]:.2f},1,1,{conf:.4f}\n"
                    )

        result_file.close()
