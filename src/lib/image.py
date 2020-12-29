
import cv2
import os

class Image:
    def __init__(self):
        self.im = None

    def set_image(self, im):
        self.im =im
        
    def get_image(self):
        return self.im

    def load_image_from_filepath(self, filepath):
        self.im = cv2.imread(filepath)

    def write_to(self, filepath):
        cv2.imwrite(filepath, self.im)

    def resize_sqaure_with_padding(self):
        (h, w) = self.im.shape[:2]
        longest_edge = max(h, w)
        top = 0
        bottom = 0
        left = 0
        right = 0
        if h < longest_edge:
            diff_h = longest_edge - h
            top = diff_h // 2
            bottom = diff_h - top
        elif w < longest_edge:
            diff_w = longest_edge - w
            left = diff_w // 2
            right = diff_w - left
        else:
            pass
        self.im = cv2.copyMakeBorder(self.im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def transform_image_for_predict_with(self, px):
        _im = self.resize_sqaure_with_padding(self.im)
        _im = cv2.resize(_im, dsize=(px, px))
        self.im = _im
