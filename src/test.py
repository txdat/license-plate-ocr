import sys
import re
import numpy as np
import cv2
from lp_detection import ONNXDetector
from paddleocr.onnx_paddleocr import ONNXPaddleOcr

det = ONNXDetector()
ocr = ONNXPaddleOcr(use_angle_cls=True)


def merge_text(text):
    return " ".join([re.sub("[^a-zA-Z0-9.-]", "", t).upper() for t in text])


img = cv2.imread(sys.argv[1])
for plate_det in det(img):
    plate_img = plate_det["img"]

    plate_text = []
    for box, res in ocr.ocr(plate_det["img"])[0]:
        plate_text.append(res[0])
        rec = np.asarray(box).astype(int)
        cv2.rectangle(plate_img, rec[0], rec[2], (0, 255, 0), 2)

    print(merge_text(plate_text))

    cv2.imshow("plate_img", plate_img)
    cv2.waitKey(0)
