import copy
import os
import traceback
import cv2
from tqdm import tqdm
import numpy as np

from . import fs
from .image import Image

INBOX_FOLDER_NAME = "_others_"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class MovieClipper:
    def __init__(self, movie_path, skip, predictor):
        self.movie_path = movie_path
        self.skip = skip or 3
        self.predictor = predictor

    @property
    def output_dir(self):
        pass

    @property
    def class_mapping(self):
        pass

    @property
    def threshold(self):
        pass

    @output_dir.setter
    def output_dir(self, output_dir):
        self.__output_dir = output_dir

    @class_mapping.setter
    def class_mapping(self, class_mapping):
        self.__class_mapping = class_mapping

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    def caching_to(self, cache_filepath):
        key = fs.get_filename_without_ext(self.movie_path)
        with open(cache_filepath, mode='a') as f:
            f.write(f"\n{key}")

    def is_completed_clip(self, cache_filepath):
        bool = False

        if not os.path.isfile(cache_filepath):
            return bool

        with open(cache_filepath) as f:
            lines = f.readlines()
            key = fs.get_filename_without_ext(self.movie_path)
            if key in lines:
                bool = True

        return bool

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
        crop_image_path = os.path.join(self.__output_dir, name, fs.add_prefix_to('{}.png'.format(name)))
        os.makedirs(os.path.dirname(crop_image_path), exist_ok=True)
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", crop_image_path)

    def capture(self):
        cap = cv2.VideoCapture(self.movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0
        pbar = tqdm(range(start_pos, frame_count, int(fps / fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0:
                pbar.update(1)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            if frame is None:
                pbar.update(1)
                continue

            crop = self.crop_name_area(frame)
            if crop is None:
                pbar.update(1)
                break

            try:
                image = Image()
                image.set_image(crop)
                pred_class, pred_proba = self.predictor.predict(image)
                print(pred_class, pred_proba)
                if crop is None:
                    pbar.update(1)
                    break

                name = INBOX_FOLDER_NAME
                if pred_proba >= self.__threshold:
                    name = self.__class_mapping.get(str(pred_class))
                print(name)
                self.clip_frame(crop, name)
            except Exception as e:
                print(traceback.format_exc())
                pbar.update(1)
                continue
        cap.release()
