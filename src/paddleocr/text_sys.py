import os
import cv2
import copy
from .text_det import TextDetector
from .text_cls import TextClassifier
from .text_rec import TextRecognizer
from .utils import get_rotate_crop_image, get_minarea_rect_crop, sorted_boxes


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = TextDetector(args)
        self.text_recognizer = TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )

        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        ori_im = img.copy()
        # 文字检测
        dt_boxes = self.text_detector(img)

        if dt_boxes is None:
            return None, None

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # 图片裁剪
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # 方向分类
        if self.use_angle_cls and cls:
            img_crop_list, angle_list = self.text_classifier(img_crop_list)

        # 图像识别
        rec_res = self.text_recognizer(img_crop_list)

        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res
