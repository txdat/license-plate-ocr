import sys
import re
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from lp_detection import ONNXDetector
from paddleocr.onnx_paddleocr import ONNXPaddleOcr

det = ONNXDetector()
paddle_ocr = ONNXPaddleOcr(use_angle_cls=False)


def merge_text(text):
    text = [re.sub("[^a-zA-Z0-9]", "", t).upper() for t in text]
    text = [t for t in text if t[:1].isdigit()]
    return "".join(text)


def preprocess_ocr(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def predict(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        results = []
        for plate_det in det(img):
            plate_img = plate_det["img"]
            # plate_img = preprocess_ocr(plate_img)
            plate_res = []
            for box, box_res in paddle_ocr.ocr(plate_img)[0]:
                box = np.asarray(box).astype(int).tolist()
                plate_res.append(
                    {"box": [*box[0], *box[2]], "text": box_res[0], "conf": box_res[1]}
                )
            plate_res = sorted(plate_res, key=lambda x: x["box"][1])
            results.append((plate_img, merge_text([r["text"] for r in plate_res])))

        return results
    except Exception as e:
        return []


def fix_jpeg_sos_parameters(image_folder):
    # Iterate through all files in the directory
    for filename in os.listdir(image_folder):
        # Process only JPEG files
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(image_folder, filename)
            try:
                # Open the image
                with Image.open(image_path) as img:
                    # If the image opens without errors, re-save it
                    img.save(image_path, "JPEG")
                    print(
                        f"Re-saved image {filename} successfully to ensure proper SOS parameters."
                    )
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    from tqdm import tqdm

    # img_paths = []
    # texts = []
    # i = 0
    # for path in tqdm(os.listdir("../data/LP")):
    #     if path.endswith(".txt"):
    #         continue
    #     image_path = f"../data/LP/{path}"
    #     results = predict(image_path)
    #     for j, (img, text) in enumerate(results):
    #         img_path = f"../data/LP_ocr/{i}_{j}.jpg"
    #         img_paths.append(img_path)
    #         texts.append(text)
    #         cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    #     i += 1

    # df = pd.DataFrame()
    # df["img_path"] = img_paths
    # df["text"] = texts

    # df.to_csv("/tmp/test.csv", sep="\t")

    df = pd.read_csv("~/Downloads/lp-test.csv")

    texts = []
    for path in tqdm(df["image"]):
        image_path = f"../data/LP/{path}"
        results = predict(image_path)
        if len(results) == 0:
            texts.append("")
            continue
        texts.append(results[0][1])

    df["predict"] = texts
    df.to_csv("/tmp/test2.csv", sep="\t")
