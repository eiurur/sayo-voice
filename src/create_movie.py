from __future__ import print_function
import sys
import cv2
import os
import shutil
import traceback
import joblib
from tqdm import tqdm
from datetime import datetime
from multiprocessing import current_process, Pool, Process
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

from lib import fs
from lib.image import Image

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
JOB_NUM = 3

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
movie_dir = os.path.join(cwd.parent, 'movies')
record_dir = os.path.join(cwd, "02.clip", "records")
crop_dir = os.path.join(cwd, "03.movie", "crop_movies")
tmp_dir = os.path.join(cwd, "03.movie", "crop_tmp")

def clip_movie(movie_file_path, clips, dst):
    print(movie_file_path, clips, dst)
    video = VideoFileClip(movie_file_path)
    clipsArray = [] 
    for clip in clips:
        [startTime, endTime] = list(map(int, clip.split('-')))
        print(startTime, endTime)
        clip = video.subclip(startTime / video.fps, endTime / video.fps)
        clipsArray.append(clip)
    final = concatenate_videoclips(clipsArray)
    final.write_videofile(dst, fps=video.fps, codec='libx264', audio_codec="aac")

def process(src_record_path, chara_crop_dir, chara_tmp_dir):
    try:
        with open(src_record_path, encoding='UTF-8') as f:
            lines = f.readlines()
            movie_raw_file_path = lines[0].strip()
            movie_encoded_file_name = lines[1].strip()

            clips = list(map(lambda x: x.strip(), lines[2:]))
            if len(clips) == 0: 
                return

            dst = os.path.join(chara_crop_dir, movie_encoded_file_name)
            if os.path.exists(dst): 
                return

            tmp_file_path = os.path.join(chara_tmp_dir, movie_encoded_file_name)
            shutil.copy2(movie_raw_file_path, tmp_file_path)
            clip_movie(tmp_file_path, clips, dst)
    except Exception as e:
        print("process ERROR: ", src_record_path)
        print(e)
        print(traceback.format_exc())

def prepare(charactor_name, series_name):
    chara_crop_dir = os.path.join(crop_dir, charactor_name, series_name)
    if not os.path.exists(chara_crop_dir):
        os.makedirs(chara_crop_dir, exist_ok=True)

    chara_tmp_dir = os.path.join(tmp_dir, charactor_name, series_name)
    if not os.path.exists(chara_tmp_dir):
        os.makedirs(chara_tmp_dir, exist_ok=True)

    return chara_crop_dir, chara_tmp_dir
    
def main():
    for series_name in CLIP_TARGET_FOLDER_NAMES:
        charactor_dirs = fs.list_dirs(record_dir)
        for charactor_name in tqdm(charactor_dirs):
            chara_crop_dir, chara_tmp_dir = prepare(charactor_name, series_name)
            chara_record_dir = os.path.join(record_dir, charactor_name, series_name)
            record_pathes = fs.list_files(chara_record_dir)
            joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
                src_record_path=record_path,
                chara_crop_dir=chara_crop_dir,
                chara_tmp_dir=chara_tmp_dir
            ) for record_path in record_pathes])


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
    