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
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process

#print('numpy:', np.__version__)
#print('pandas:', pd.__version__)
#print('matplotlib:', mplv)
print('OpenCV:', cv2.__version__)

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
CHARACTOR_NAMES = ["sayo", "tsugumi", "tae", "kanon", "hina", "aya", "eve"]
THRESHOLD = 55.0
COMPARE_IMAGE_NAME_FORMAT = "name-{}.png"


def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]

def get_filename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]

def get_filename_and_ext(p):
    return os.path.splitext(os.path.basename(p))

class Movie:
    def __init__(self, src_movie_path, skip, records):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.records = records

    def capture(self, pid):
        cwd = os.getcwd()
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0  # fps:  fps * (60 * 20) .. 20分

        pbar = tqdm(range(start_pos, frame_count, int(fps/fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip == 0: continue 
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ret, frame = cap.read()
            if frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (height, width) = rgb.shape[:2]
            if height != 1080 and width != 1920:
                break

            # print(height, width)
            crop = rgb[780:850, 230:450] # 1920 x 1080:  rgb[775:850, 200:660]　, [790:840, 230:350]
            """
            serif = rgb[780:1050, 200:1600]
            plt.gray()
            plt.imshow(crop)
            plt.imshow(serif)
            plt.show()
            """
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
    def __init__(self, record_dir_format, name, pid):
        self.record_dir_format = record_dir_format
        self.name = name
        self.pid = pid
        self.periods = []
        self.start_frame = -1
    
    def prepare(self):
        dir_path = self.record_dir_format.format(self.name)
        print(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def compare(self, frame_idx, crop):
        cwd = os.getcwd()
        crop_image_path = os.path.join(cwd, 'temp_{}_{}.png'.format(self.name, self.pid))
        name_image_path = os.path.join(cwd, 'compares', COMPARE_IMAGE_NAME_FORMAT.format(self.name))

        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", pid, name)
            print(traceback.format_exc())
            return False

        target_img = cv2.imread(crop_image_path, cv2.IMREAD_GRAYSCALE)
        comparing_img = cv2.imread(name_image_path, cv2.IMREAD_GRAYSCALE)
        detector = cv2.ORB_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)
        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(target_des, comparing_des)
        matches = sorted(matches, key=lambda x: x.distance)
        dist = [m.distance for m in matches]
        if len(dist) == 0: return False

        ret = sum(dist) / len(dist)
        # print(" ret - {}: {}".format(self.name, ret))
        if self.start_frame == -1 and ret <= THRESHOLD:
            self.start_frame = frame_idx
        if self.start_frame != -1 and ret > THRESHOLD:
            end_frame = frame_idx
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1

    def write(self, prefix_data, movie_file_name_without_ext):
        if prefix_data is None:
            prefix_data = []
        dir_path = self.record_dir_format.format(self.name)
        record_file = os.path.join(dir_path, "{}.txt".format(movie_file_name_without_ext))
        if os.path.exists(record_file): return

        data = prefix_data + self.periods
        with open(record_file, "w", encoding='UTF-8') as f:
            f.write('\n'.join(data))


def sample_func(src_movie_path, record_dir_format):
    try:
        sleep(0.01)
        cp = current_process()
        pid = cp._identity[0]
        print(cp._identity)
        records = [Record(record_dir_format, name, pid) for name in CHARACTOR_NAMES]
        for record in records:
            record.prepare()
        # map(lambda record: record.prepare(), records)
        movie = Movie(src_movie_path, 3, records)
        movie.capture(pid)
        movie.write_period_to_file()
    except Exception as e:
        print("SAMPLE_FUNC ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())

def main():
    cwd = os.getcwd()
    movie_dir = os.path.join(cwd, 'movies')

    for series_name in CLIP_TARGET_FOLDER_NAMES:
        record_dir_format = os.path.join(cwd, "records", "{}", series_name)
        movies = list_files(os.path.join(movie_dir, series_name))
        pbar = tqdm(movies)
        for p in pbar:
            params = [(movie, record_dir_format) for movie in movies]
            with Pool(processes=4) as pool:
                results = [pool.apply_async(sample_func, param) for param in params]
                for r in results:
                    print('\t', r.get())

if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
