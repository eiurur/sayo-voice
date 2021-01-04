#!/usr/bin/env python
# coding: utf-8

import sys
import os
import traceback
import joblib
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from lib import fs
from lib.image import Image
from lib.record import Record
from lib.movie_predict_clipper import MoviePredictClipper

import tensorflow as tf
from tensorflow import keras

# $CLIP_TARGET_FOLDER_NAMES = ["s2"]
CLIP_TARGET_FOLDER_NAMES = ["s2", "band"]
OUTPUT_FOLDER_NAME = "classification"
CACHE_FILE_NAME = "classification_cache.txt"
JOB_NUM = 4
THRESHOLD = 95.0
IMAGE_SIZE_PX = 112
SKIP_FRAME_INTEVAL = 60

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "00.dataset")
cache_filepath = os.path.join(resource_dir, CACHE_FILE_NAME)
clip_output_dir = os.path.join(resource_dir, OUTPUT_FOLDER_NAME)
movie_dir = os.path.join(cwd.parent, "assets", 'movies')
class_mapping_file_path = os.path.join(cwd.parent, "model", "class_mapping.json")
model_file_path = os.path.join(cwd.parent, "model", "model.h5")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def process(src_movie_path, class_mapping):
    try:
        model = keras.models.load_model(model_file_path, compile=False)
        movie_predict_clipper = MoviePredictClipper(src_movie_path, SKIP_FRAME_INTEVAL, 0)
        movie_predict_clipper.model = model
        movie_predict_clipper.threshold = THRESHOLD
        movie_predict_clipper.px = IMAGE_SIZE_PX
        movie_predict_clipper.output_dir = clip_output_dir
        movie_predict_clipper.class_mapping = class_mapping
        if movie_predict_clipper.isCompltedClip(cache_filepath):
            return
        movie_predict_clipper.capture(0)
        movie_predict_clipper.caching_to(cache_filepath)
    except Exception as e:
        print("process ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    class_mapping = fs.load_json(class_mapping_file_path)
    for series_name in tqdm(CLIP_TARGET_FOLDER_NAMES):
        movie_pathes = fs.list_entries(os.path.join(movie_dir, series_name))
        joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
            src_movie_path=movie_path,
            class_mapping=class_mapping
        ) for movie_path in movie_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
