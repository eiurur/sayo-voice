#!/usr/bin/env python
# coding: utf-8

import os
import traceback
import joblib
from pathlib import Path
from datetime import datetime

from lib import fs
from lib.image import Image
from lib.record import Record
from lib.movie import Movie

import tensorflow as tf
from tensorflow import keras

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2", "band"]
JOB_NUM = 3
THRESHOLD = 99.0
IMAGE_SIZE_PX = 112
SKIP_FRAME_INTEVAL = 3

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "02.record")
movie_dir = os.path.join(cwd.parent, "assets", 'movies')
class_mapping_file_path = os.path.join(cwd.parent, "model", "class_mapping.json")
model_file_path = os.path.join(cwd.parent, "model", "model.h5")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def process(src_movie_path, record_dir_format, class_mapping):
    try:
        records = []
        for label in class_mapping.items():
            model = keras.models.load_model(model_file_path, compile=False)
            record = Record()
            record.dir_format = record_dir_format
            record.label = label
            record.threshold = THRESHOLD
            record.skip_frame_interval = SKIP_FRAME_INTEVAL
            record.model = model
            record.px = IMAGE_SIZE_PX
            record.prepare()
            records.append(record)
        movie = Movie(src_movie_path, SKIP_FRAME_INTEVAL, records)
        if movie.isCompltedClip():
            return
        movie.capture()
        movie.write_period_to_file()
    except Exception as e:
        print("record ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    class_mapping = fs.load_json(class_mapping_file_path)
    for series_name in CLIP_TARGET_FOLDER_NAMES:
        record_dir_format = os.path.join(resource_dir, "records", "{}", series_name)
        movies = fs.list_entries(os.path.join(movie_dir, series_name))
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
