from pathlib import Path
from ultralytics import YOLO

module_dir = Path(__file__).resolve().parent


class ONNXDetector(object):
    def __init__(self, model_type="yolov11n"):
        self.det = YOLO(f"{module_dir}/models/{model_type}.onnx", task="detect")

    def __call__(self, image):
        h, w, _ = image.shape
        dets = []
        for det in self.det(image, verbose=False):
            boxes = det.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                [x1, y1, x2, y2] = box
                dets.append({"img": image[y1:y2, x1:x2], "box": box})
        return dets
