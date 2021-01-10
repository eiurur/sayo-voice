#!/usr/bin/env python
# coding: utf-8

import os
import traceback
import joblib
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from lib import fs
from lib.movie_clipper import MovieClipper

from tensorflow import keras

CLIP_TARGET_FOLDER_NAMES = ["s2", "band"]
JOB_NUM = 1
THRESHOLD = 99.0
SKIP_FRAME_INTEVAL = 60

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "00.dataset")
cache_filepath = os.path.join(resource_dir, "classification_cache.txt")
clip_output_dir = os.path.join(resource_dir, "classification")
movie_dir = os.path.join(cwd.parent, "assets", 'movies')
class_mapping_file_path = os.path.join(cwd.parent, "assets", "class", "class_mapping.json")
model_file_path = os.path.join(cwd.parent, "assets", "class", "model.h5")
config_file_path = os.path.join(cwd.parent, "config.json")

config_file = open(config_file_path, 'r')
config = json.load(config_file)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def process(movie_path, class_mapping):
    try:
        model = keras.models.load_model(model_file_path, compile=False)
        movie_clipper = MovieClipper(movie_path, SKIP_FRAME_INTEVAL, 0)
        movie_clipper.model = model
        movie_clipper.threshold = THRESHOLD
        movie_clipper.px = config["image_size_px"]
        movie_clipper.output_dir = clip_output_dir
        movie_clipper.class_mapping = class_mapping
        if movie_clipper.is_completed_clip(cache_filepath):
            return
        movie_clipper.capture(0)
        movie_clipper.caching_to(cache_filepath)
    except Exception as e:
        print("process ERROR: ", movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    class_mapping = fs.load_json(class_mapping_file_path)
    for series_name in tqdm(CLIP_TARGET_FOLDER_NAMES):
        movie_pathes = fs.list_entries(os.path.join(movie_dir, series_name))
        joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
            movie_path=movie_path,
            class_mapping=class_mapping
        ) for movie_path in movie_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
