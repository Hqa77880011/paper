import cv2
import numpy as np
from pycocotools import mask as coco_mask

class DisplayUtils:
    def __init__(self):
        self.transparency = 0.3
        self.box_width = 2
        self.text_size = 1.5  # 默认文字大小

    def increase_transparency(self):
        self.transparency = min(1.0, self.transparency + 0.05)
    
    def decrease_transparency(self):
        self.transparency = max(0.0, self.transparency - 0.05)

    def increase_text_size(self):
        self.text_size = min(5.0, self.text_size + 0.1)  # 最大文字大小为 5.0

    def decrease_text_size(self):
        self.text_size = max(0.5, self.text_size - 0.1)  # 最小文字大小为 0.5

    def overlay_mask_on_image(self, image, mask, color=(0, 0, 255)):
        gray_mask = mask.astype(np.uint8) * 255
        gray_mask = cv2.merge([gray_mask, gray_mask, gray_mask])
        color_mask = cv2.bitwise_and(gray_mask, color)
        masked_image = cv2.bitwise_and(image.copy(), color_mask)
        overlay_on_masked_image = cv2.addWeighted(
            masked_image, self.transparency, color_mask, 1 - self.transparency, 0
        )
        background = cv2.bitwise_and(image.copy(), cv2.bitwise_not(color_mask))
        image = cv2.add(background, overlay_on_masked_image)
        return image

    def __convert_ann_to_mask(self, ann, height, width):
        mask = np.zeros((height, width), dtype=np.uint8)
        poly = ann["segmentation"]
        rles = coco_mask.frPyObjects(poly, height, width)
        rle = coco_mask.merge(rles)
        mask_instance = coco_mask.decode(rle)
        mask_instance = np.logical_not(mask_instance)
        mask = np.logical_or(mask, mask_instance)
        mask = np.logical_not(mask)
        return mask

    def draw_box_on_image(self, image, categories, ann, color):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        image = cv2.rectangle(image, (x, y), (x + w, y + h), color, self.box_width)

        text = '{} {}'.format(ann["id"],categories[ann["category_id"]])
        txt_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, self.text_size, 1)[0]
        thickness = max(1, int(self.text_size * 3.33))

        cv2.rectangle(image, (x, y + 1), (x + txt_size[0] + 1, y + int(self.text_size*txt_size[1])), color, -1)
        cv2.putText(image, text, (x, y + txt_size[1]), font, self.text_size, txt_color, thickness=thickness)
        return image

    def draw_annotations(self, image, categories, annotations, colors):
        for ann, color in zip(annotations, colors):
            image = self.draw_box_on_image(image, categories, ann, color)
            mask = self.__convert_ann_to_mask(ann, image.shape[0], image.shape[1])
            image = self.overlay_mask_on_image(image, mask, color)
        return image

    def draw_points(
        self, image, points, labels, colors={1: (0, 255, 0), 0: (0, 0, 255)}, radius=5
    ):
        for i in range(points.shape[0]):
            point = points[i, :]
            label = labels[i]
            color = colors[label]
            image = cv2.circle(image, tuple(point), radius, color, -1)
        return image
