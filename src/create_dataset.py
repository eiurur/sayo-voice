#!/usr/bin/env python
# coding: utf-8

import sys
import cv2
import os
import traceback
import joblib
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process
from pathlib import Path

from lib import fs
from lib.image import Image

CLIP_TARGET_FOLDER_NAMES = ["s2"]
# CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
OUTPUT_FOLDER_NAME = "inbox"
CACHE_FILE_NAME = "cache.txt"
THRESHOLD = 55.0
JOB_NUM = 2

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
output_dir = os.path.join(cwd, "00.dataset")


class MovieClipper:
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
        
    def crop_name_area(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height == 1080 and width == 1920:
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            return rgb[0:0, 0:0]
        return None

    def clip_frame(self, frame_idx, crop):
        crop_image_path = os.path.join(output_dir, OUTPUT_FOLDER_NAME, fs.add_prefix_to('{}.png'.format(self.pid)))
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

            crop = self.crop_name_area(frame)
            if crop is None: break

            self.clip_frame(frame_idx, crop)
        cap.release()

def process(src_movie_path):
    try:
        sleep(0.01)
        cp = current_process()
        pid = cp._identity[0]
        print(cp._identity)
        movie = MovieClipper(src_movie_path, 180, pid)
        cache_filepath = os.path.join(output_dir, CACHE_FILE_NAME)
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
        movie_pathes = fs.list_files(os.path.join(movie_dir, series_name))
        pbar = tqdm(movie_pathes)
        for p in pbar:
            export_data_list = joblib.Parallel(n_jobs=JOB_NUM)(
                [joblib.delayed(process)(src_movie_path=movie_path) for movie_path in movie_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
