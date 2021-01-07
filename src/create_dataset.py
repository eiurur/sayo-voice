#!/usr/bin/env python
# coding: utf-8

import os
import traceback
import joblib
from tqdm import tqdm
from datetime import datetime
from multiprocessing import current_process
from pathlib import Path

from lib import fs
from lib.movie_clipper import MovieClipper


CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
JOB_NUM = 2
SKIP_FRAME_INTEVAL = 180

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "00.dataset")
cache_filepath = os.path.join(resource_dir, "cache.txt")
clip_output_dir = os.path.join(resource_dir, "inbox")
movie_dir = os.path.join(cwd.parent, "assets", 'movies')


def process(src_movie_path):
    try:
        cp = current_process()
        pid = cp._identity[0]
        movie_clipper = MovieClipper(src_movie_path, SKIP_FRAME_INTEVAL, pid)
        movie_clipper.output_dir = clip_output_dir
        if movie_clipper.isCompltedClip(cache_filepath):
            return
        movie_clipper.capture(pid)
        movie_clipper.caching_to(cache_filepath)
    except Exception as e:
        print("process ERROR: ", src_movie_path)
        print(e)
        print(traceback.format_exc())


def main():
    for series_name in tqdm(CLIP_TARGET_FOLDER_NAMES):
        movie_pathes = fs.list_entries(os.path.join(movie_dir, series_name))
        joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
            src_movie_path=movie_path
        ) for movie_path in movie_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
