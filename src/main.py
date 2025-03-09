from flask import Flask, request, jsonify
import numpy as np
import cv2
import re
from lp_detection import ONNXDetector
from paddleocr.onnx_paddleocr import ONNXPaddleOcr

det = ONNXDetector()
paddle_ocr = ONNXPaddleOcr(use_angle_cls=False)

app = Flask(__name__)


def merge_text(text):
    text = [re.sub("[^a-zA-Z0-9.-]", "", t).upper() for t in text]
    text = [t for t in text if t[:1].isdigit()]
    return " ".join(text)


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_UNCHANGED)

        res = []
        for plate_det in det(img):
            plate_res = []
            for box, box_res in paddle_ocr.ocr(plate_det["img"])[0]:
                box = np.asarray(box).astype(int).tolist()
                plate_res.append(
                    {"box": [*box[0], *box[2]], "text": box_res[0], "conf": box_res[1]}
                )
            res.append(
                {
                    "plate_box": plate_det["box"].tolist(),
                    "plate_res": plate_res,
                    "plate_text": merge_text([r["text"] for r in plate_res]),
                }
            )

        return jsonify({"code": 200, "message": "OK", "res": res})

    except Exception as e:
        return jsonify({"code": 500, "message": str(e), "res": []})


if __name__ == "__main__":
    app.run("0.0.0.0", port=8080, debug=False)
