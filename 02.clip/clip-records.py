#!/usr/bin/env python
# coding: utf-8


import sys
import os
import math
import glob
import traceback
import joblib
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process
from ..lib.image import Image
from ..lib.record import Record
from ..lib.movie import Movie
from ..lib import fs

# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
JOB_NUM = 1
THRESHOLD = 55.0
IMAGE_SIZE_PX = 112

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
class_mapping_file_path = os.path.join(cwd.parent, "model", "class_mapping.json")
model_file_path = os.path.join(cwd.parent, "model", "dnn_model.h5")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
model = keras.models.load_model(model_file_path, compile=False)

def process(src_movie_path, record_dir_format, class_mapping):
    try:
        sleep(0.01)
        records = []
        for label in class_mapping.items():
            record = Record()
            record.dir_format = record_dir_format
            record.label = label
            record.threshold = THRESHOLD
            record.model = model
            record.prepare()
            records.append(record)
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
    print(cwd.parent)
    print(class_mapping_file_path)
    movie_dir = os.path.join(cwd.parent, 'movies')
    class_mapping = fs.load_json(class_mapping_file_path)

    for series_name in CLIP_TARGET_FOLDER_NAMES:
        record_dir_format = os.path.join(cwd, "records", "{}", series_name)
        movies = fs.list_files(os.path.join(movie_dir, series_name))
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
