#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function
import sys
import cv2
import os
import math
import glob
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import traceback
import joblib
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process


# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

print('OpenCV:', cv2.__version__)

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
class_mapping_file_path = os.path.join(cwd, "class_mapping.json")
print(cwd.parent)
print(class_mapping_file_path)

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
JOB_NUM = 1
THRESHOLD = 55.0
IMAGE_SIZE_PX = 112
MODEL_NAME = "dnn_model.h5"

# TensorFlow GPUメモリの割合を制限する
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model = keras.models.load_model(MODEL_NAME, compile=False)

def resize_with_padding(img):
    (h, w) = img.shape[:2]
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
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]


def get_filename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]


def get_filename_and_ext(p):
    return os.path.splitext(os.path.basename(p))

def write_json(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load_json(file_path):
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        return json_data
    return None

def transform_image_for_predict_with(im):
    _im = resize_with_padding(im)
    _im = cv2.resize(_im, dsize=(IMAGE_SIZE_PX, IMAGE_SIZE_PX))
    to = os.path.join("aaa.jpg")
    cv2.imwrite(to, _im)
    return _im

class Movie:
    def __init__(self, src_movie_path, skip, records):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.records = records

    def isCompltedClip(self):
        bool = False
        for record in self.records:
            dir_path = record.record_dir_format.format(record.label_name)
            movie_file_name_without_ext = os.path.splitext(
                os.path.basename(self.src_movie_path))[0]
            record_file = os.path.join(
                dir_path, "{}.txt".format(movie_file_name_without_ext))
            if os.path.exists(record_file):
                bool = True
                break
        return bool

    def crop(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        """
        serif = rgb[780:1050, 200:1600]
        plt.gray()
        plt.imshow(crop)
        plt.imshow(serif)
        plt.show()
        """
        if height == 1080 and width == 1920:
            # 1920 x 1080:  rgb[775:850, 200:660], [790:840, 230:350]
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            return rgb[0:0, 0:0]
        return None

    def capture(self):
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0  # fps:  fps * (60 * 20) .. 20分

        pbar = tqdm(range(start_pos, frame_count, int(fps/fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ret, frame = cap.read()
            if frame is None:
                continue

            crop = self.crop(frame)
            if crop is None:
                break

            for record in self.records:
                record.compare(frame_idx, crop)
        cap.release()

    def write_period_to_file(self):
        for record in self.records:
            [filename, ext] = get_filename_and_ext(self.src_movie_path)
            hs = hashlib.md5(filename.encode()).hexdigest()
            ascii_filename = "{}{}".format(hs, ext)
            prefix_data = [self.src_movie_path, ascii_filename]
            movie_file_name_without_ext = os.path.splitext(os.path.basename(self.src_movie_path))[0]
            record.write(prefix_data, movie_file_name_without_ext)


class Record:
    def __init__(self, record_dir_format, label_i, label_name):
        self.record_dir_format = record_dir_format
        self.label_i = label_i
        self.label_name = label_name
        self.periods = []
        self.start_frame = -1

    def prepare(self):
        dir_path = self.record_dir_format.format(self.label_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def compare(self, frame_idx, crop):
        try:
            crop_with_padding = transform_image_for_predict_with(crop)
        except Exception as e:
            print(traceback.format_exc())
            return False

        img_predict = [crop_with_padding]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1) # 2次元配列を1次元に変換
        data_predict = data_predict.astype('float32')   # int型をfloat32型に変換
        data_predict /= 255                        # [0-255]の値を[0.0-1.0]に変換

        #result_predict = model.predict(data_predict)
        #result_predict_classes = model.predict_classes(data_predict)
        #result_predict_proba = model.predict_proba(data_predict)
        #cur_class = result_predict_classes[0]
        #cur_proba = result_predict_proba[0]
        predictions = model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        print(pred_class, pred_proba)
        if self.start_frame == -1 and (pred_proba >= THRESHOLD and str(self.label_i) == str(pred_class)):
            self.start_frame = frame_idx
        if self.start_frame != -1 and (pred_proba < THRESHOLD or str(self.label_i) != str(pred_class)):
            end_frame = frame_idx
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            print(pred_class, pred_proba)
            print(self.label_name, self.periods)

    def write(self, prefix_data, movie_file_name_without_ext):
        if prefix_data is None:
            prefix_data = []
        dir_path = self.record_dir_format.format(self.label_name)
        record_file = os.path.join(
            dir_path, "{}.txt".format(movie_file_name_without_ext))
        if os.path.exists(record_file):
            return

        data = prefix_data + self.periods
        with open(record_file, "w", encoding='UTF-8') as f:
            f.write('\n'.join(data))


def process(src_movie_path, record_dir_format, class_mapping):
    try:
        sleep(0.01)
        records = [Record(record_dir_format, label_i, label_name) for label_i, label_name in class_mapping.items()]
        for record in records:
            record.prepare()
        # map(lambda record: record.prepare(), records)
        movie = Movie(src_movie_path, 4, records)
        if movie.isCompltedClip():
            return

        movie.capture()
        movie.write_period_to_file()
    except Exception as e:
        print("SAMPLE_FUNC ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    movie_dir = os.path.join(cwd.parent, 'movies')
    class_mapping = load_json(class_mapping_file_path)

    for series_name in CLIP_TARGET_FOLDER_NAMES:
        record_dir_format = os.path.join(cwd, "records", "{}", series_name)
        movies = list_files(os.path.join(movie_dir, series_name))
        pbar = tqdm(movies)
        for p in pbar:
            params = [(movie, record_dir_format) for movie in movies]
            joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
                src_movie_path=param[0], 
                record_dir_format=param[1], 
                class_mapping=class_mapping
            ) for param in params])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
