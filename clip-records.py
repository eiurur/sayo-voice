#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function
import sys
import cv2
import os
import math
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import threading
from multiprocessing import current_process, Pool, Process

#print('numpy:', np.__version__)
#print('pandas:', pd.__version__)
#print('matplotlib:', mplv)
print('OpenCV:', cv2.__version__)


THRESHOLD = 50.0


def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]


def capture(id, p):
    records = [p]

    cwd = os.getcwd()
    name_file = os.path.join(cwd, 'compares', 'name-short.png')

    cap = cv2.VideoCapture(p)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("PATH: ", p)
    print("FRAME_COUNT: ", frame_count)
    print("FPS: ", fps)

    start_frame = -1
    start_pos = 0  # fps:  fps * (60 * 20) .. 20分

    pbar = tqdm(range(start_pos, frame_count, int(fps/fps)))
    for idx in pbar:
        if idx % 2 == 0: continue # fps == 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        ret, frame = cap.read()
        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height != 1080 and width != 1920:
            break

        # print(height, width)
        # 1920 x 1080:  rgb[775:850, 200:660]　, [790:840, 230:350]
        crop = rgb[780:850, 230:450]
        # 1920 x 1080:  rgb[775:850, 200:660]　, [790:840, 230:350]
        serif = rgb[780:1050, 200:1600]
        """
        plt.gray()
        plt.imshow(crop)
        plt.imshow(serif)
        plt.show()
        """
        crop_image_path = os.path.join(cwd, 'temp_{}.png'.format(id))
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", p)
            continue

        target_img = cv2.imread(crop_image_path, cv2.IMREAD_GRAYSCALE)
        comparing_img = cv2.imread(name_file, cv2.IMREAD_GRAYSCALE)
        detector = cv2.ORB_create()
        (target_kp, target_des) = detector.detectAndCompute(target_img, None)
        (comparing_kp, comparing_des) = detector.detectAndCompute(comparing_img, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(target_des, comparing_des)
        matches = sorted(matches, key=lambda x: x.distance)
        dist = [m.distance for m in matches]
        if len(dist) == 0:
            continue
        ret = sum(dist) / len(dist)
        if start_frame == -1 and ret <= THRESHOLD:
            start_frame = idx
        if start_frame != -1 and ret > THRESHOLD:
            end_frame = idx
            records.append("{}-{}".format(start_frame, end_frame))
            start_frame = -1
        # print(ret)
    cap.release()
    return records


def sample_func(p, record_dir):
    try:
        cp = current_process()
        print(cp._identity, p)
        p_without_ext = os.path.splitext(os.path.basename(p))[0]
        record_file = os.path.join(record_dir, "{}.txt".format(p_without_ext))
        if os.path.exists(record_file):
            return

        records = capture(cp._identity[0], p)
        with open(record_file, "w", encoding='UTF-8') as f:
            print(record_file, records)
            f.write('\n'.join(records))
    except Exception as e:
        print("ERROR: ", p)
        print(e)


def main():
    cwd = os.getcwd()
    movie_dir = os.path.join(cwd, 'movies')
    crop_dir = os.path.join(cwd, 'crops')
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir, exist_ok=True)

    for series_name in ['s1', 's2']:
        movies = list_files(os.path.join(movie_dir, series_name))
        record_dir = os.path.join(cwd, "records", series_name)
        if not os.path.exists(record_dir):
            os.makedirs(record_dir, exist_ok=True)

        process_list = []
        pbar = tqdm(movies)
        for p in pbar:
            params = [(movie, record_dir) for movie in movies]
            with Pool(processes=4) as pool:
                results = [pool.apply_async(sample_func, param)
                           for param in params]
                for r in results:
                    print('\t', r.get())

if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
