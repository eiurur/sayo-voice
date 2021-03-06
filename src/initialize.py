#!/usr/bin/env python
# coding: utf-8

import os
import traceback
import joblib
import youtube_dl
from datetime import datetime
from pathlib import Path


DOWNLOADING_PLAYLIST_MAP = {
    "s1": "PL_-PeRPsOsKLGsG8u6f0P5XVCANWZM6mY",
    "s2": "PL_-PeRPsOsKLs2v8TpWZfJlZ7aic8TFsg",
    "band": "PL_-PeRPsOsKLQNENQVw7E96fozDaZIs16",
    "main": "PL_-PeRPsOsKK0IHPDktxYxHRV2kzBIyEw",
    "area1": "PL_-PeRPsOsKLZH1dr2ohPfXuWIwzfuTn0",
    "area2": "PL_-PeRPsOsKLlDGZWV72ukYtV1c50mw7O"
}
JOB_NUM = 3

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
movie_dir = os.path.join(cwd.parent, "assets", 'movies')


def download(dir_name, playlist_name):
    dst_dir = os.path.join(movie_dir, dir_name)
    ydl_opts = {
        "outtmpl": "{VIDEO_DIR}/%(title)s.mp4".format(VIDEO_DIR=dst_dir),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    }
    print("Downloading {name} start..".format(name=playlist_name))
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(
            playlist_name,
            download=True
        )
    print("Downloading {name} finish!".format(name=playlist_name))


def process(dir_name, playlist_name):
    try:
        download(dir_name, playlist_name)
    except Exception as e:
        print(traceback.format_exc())


def main():
    joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
        dir_name=dir_name,
        playlist_name=playlist_name
    ) for dir_name, playlist_name in DOWNLOADING_PLAYLIST_MAP.items()])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
