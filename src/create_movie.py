from __future__ import print_function
import sys
import cv2
import os
import shutil
from tqdm.notebook import tqdm
from datetime import datetime
from multiprocessing import Pool
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

from lib import fs
from lib.image import Image

CLIP_TARGET_FOLDER_NAMES = ["s1", "s2"]
CHARACTOR_NAMES = ["sayo", "tsugumi", "tae", "kanon", "hina", "aya", "eve"]

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

def main():
    for series_name in CLIP_TARGET_FOLDER_NAMES:
        charactor_dirs = fs.list_dirs(record_dir)
        for charactor_name in charactor_dirs:
            movies = fs.list_files(os.path.join(movie_dir, series_name))
            chara_record_dir = os.path.join(record_dir, charactor_name, series_name)
            records = fs.list_files(chara_record_dir)

            chara_crop_dir = os.path.join(crop_dir, charactor_name, series_name)
            if not os.path.exists(chara_crop_dir):
                os.makedirs(chara_crop_dir, exist_ok=True)

            chara_tmp_dir = os.path.join(tmp_dir, charactor_name, series_name)
            if not os.path.exists(chara_tmp_dir):
                os.makedirs(chara_tmp_dir, exist_ok=True)

            pbar = tqdm(records)
            for p in pbar:
                with open(p, encoding='UTF-8') as f:
                    lines = f.readlines()
                    movie_raw_file_path = lines[0].strip()
                    movie_encoded_file_name = lines[1].strip()

                    clips = list(map(lambda x: x.strip(), lines[2:]))
                    if len(clips) == 0: 
                        pbar.update(1)
                        continue

                    dst = os.path.join(chara_crop_dir, movie_encoded_file_name)
                    if os.path.exists(dst): 
                        pbar.update(1)
                        continue
                    
                    tmp_file_path = os.path.join(chara_tmp_dir, movie_encoded_file_name)
                    shutil.copy2(movie_raw_file_path, tmp_file_path)
                    clip_movie(tmp_file_path, clips, dst)

if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    