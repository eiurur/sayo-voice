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
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process
from hashlib import md5
from time import localtime
from pathlib import Path

CLIP_TARGET_FOLDER_NAMES = ["s2"]
# CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
OUTPUT_FOLDER_NAME = "inbox"
CACHE_FILE_NAME = "cache.txt"
THRESHOLD = 55.0
JOB_NUM = 2
cwd = Path(__file__).parent

def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]


def get_filename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]


def get_filename_and_ext(p):
    return os.path.splitext(os.path.basename(p))


def add_prefix_to(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"


class Movie:
    def __init__(self, src_movie_path, skip, pid):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.pid = pid

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
        
    def crop(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        # print(height, width)
        """
        serif = rgb[780:1050, 200:1600]
        plt.gray()
        plt.imshow(crop)
        plt.imshow(serif)
        plt.show()
        """
        if height == 1080 and width == 1920:
            # 1920 x 1080:  rgb[775:850, 200:660]　, [790:840, 230:350]
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            return rgb[0:0, 0:0]
        return None

    def crop_frame(self, frame_idx, crop):
        crop_image_path = os.path.join(cwd, OUTPUT_FOLDER_NAME, add_prefix_to('{}.png'.format(self.pid)))
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", pid, crop_image_path)
            print(traceback.format_exc())
            return False

    def capture(self, pid):
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0  # fps:  fps * (60 * 20) .. 20分

        pbar = tqdm(range(start_pos, frame_count, int(fps/fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0: continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ret, frame = cap.read()
            if frame is None: continue

            crop = self.crop(frame)
            if crop is None: break

            self.crop_frame(frame_idx, crop)
        cap.release()

def process(src_movie_path):
    try:
        sleep(0.01)
        cp = current_process()
        pid = cp._identity[0]
        print(cp._identity)
        movie = Movie(src_movie_path, 180, pid)
        cache_filepath = os.path.join(cwd, CACHE_FILE_NAME)
        if movie.isCompltedClip(cache_filepath): return
        movie.capture(pid)
        movie.caching_to(cache_filepath)
    except Exception as e:
        print("SAMPLE_FUNC ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    print(cwd)
    movie_dir = os.path.join(cwd.parent, 'movies')

    for series_name in CLIP_TARGET_FOLDER_NAMES:
        movie_pathes = list_files(os.path.join(movie_dir, series_name))
        pbar = tqdm(movie_pathes)
        for p in pbar:
            export_data_list = joblib.Parallel(n_jobs=JOB_NUM)(
                [joblib.delayed(process)(src_movie_path=movie_path) for movie_path in movie_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
