import copy
import math
import os
import traceback
import cv2
from tqdm import tqdm
import numpy as np

from . import fs
from .image import Image


class MoviePredictClipper:
    def __init__(self, src_movie_path, skip, pid):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.pid = pid

    @property
    def output_dir(self):
        pass

    @property
    def model(self):
        pass

    @property
    def class_mapping(self):
        pass

    @property
    def threshold(self):
        pass

    @property
    def px(self):
        pass

    @output_dir.setter
    def output_dir(self, output_dir):
        self.__output_dir = output_dir

    @model.setter
    def model(self, model):
        self.__model = model

    @class_mapping.setter
    def class_mapping(self, class_mapping):
        self.__class_mapping = class_mapping

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @px.setter
    def px(self, px):
        self.__px = px

    def get_movie_file_name(self):
        return os.path.splitext(os.path.basename(self.src_movie_path))[0]

    def caching_to(self, cache_filepath):
        key = self.get_movie_file_name()
        with open(cache_filepath, mode='a') as f:
            f.write(f"\n{key}")

    def isCompltedClip(self, cache_filepath):
        bool = False

        if not os.path.isfile(cache_filepath):
            return bool

        with open(cache_filepath) as f:
            lines = f.readlines()
            key = self.get_movie_file_name()
            if key in lines:
                bool = True

        return bool

    def crop_name_area_old(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height == 1080 and width == 1920:
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            resized = cv2.resize(rgb, (1920, 1080))
            return resized[780:850, 230:450]
            # return rgb[520:560, 150:300]
        return None

    def crop_name_area(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height == 1080 and width == 1920:
            return rgb[775:850, 220:615]
        if height == 720 and width == 1280:
            resized = cv2.resize(rgb, (1920, 1080))
            return resized[775:850, 220:615]
        return None

    def clip_frame(self, crop, name):
        crop_image_path = os.path.join(self.__output_dir, name, fs.add_prefix_to('{}.png'.format(self.pid)))
        print(crop_image_path)
        os.makedirs(os.path.dirname(crop_image_path), exist_ok=True)
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", crop_image_path)
            return False

    def capture(self, pid):
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0
        pbar = tqdm(range(start_pos, frame_count, int(fps / fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            if frame is None:
                continue

            crop_old = self.crop_name_area_old(frame)
            if crop_old is None:
                break

            try:
                image = Image()
                image.set_image(crop_old)
                pred_class, pred_proba = self.predict(image)
                if pred_proba <= self.__threshold:
                    continue

                print(pred_class, pred_proba)
                crop = self.crop_name_area(frame)
                if crop is None:
                    break
                name = self.__class_mapping.get(str(pred_class))
                print(name)
                self.clip_frame(crop, name)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                continue
        cap.release()

    def __predict(self, im):
        img_predict = [im]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], self.__px, self.__px, 1)
        data_predict = data_predict.astype('float32')
        data_predict /= 255

        predictions = self.__model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        return pred_class, pred_proba

    def predict(self, im):
        try:
            _im = copy.deepcopy(im)
            _im.transform_image_for_predict_with(self.__px)
            pred_class, pred_proba = self.__predict(_im.get_image())
            return pred_class, pred_proba
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return None, None
