import os
import traceback
import joblib
import math
import json
from tqdm import tqdm
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path

from lib import fs

JOB_NUM = 1

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "03.movie")
crop_dir = os.path.join(resource_dir, "crop_movies")
record_dir = os.path.join(cwd, "02.record", "records")
config_file_path = os.path.join(cwd.parent, "config.json")

config_file = open(config_file_path, 'r')
config = json.load(config_file)


def calc_elapsed_time(frame, fps):
    seconds = frame / fps
    hf, hi = math.modf(seconds / (60 * 60))
    h = math.floor(hi)
    mf, mi = math.modf(hf * 60)
    m = math.floor(mi)
    sf, si = math.modf(mf * 60)
    s = math.floor(si)
    ff, fi = math.modf(sf * 100)
    f = math.floor(fi)
    format = '{:0>2}:{:0>2}:{:0>2}.{:0>2}'.format(h, m, s, f)
    return format


def clip_movie(movie_file_path, clips, dst, src_record_path):
    video = VideoFileClip(movie_file_path, fps_source="fps")
    times = []
    clipsArray = []
    for clip in clips:
        [start_frame, end_frame] = list(map(int, clip.split('-')))
        start_time = calc_elapsed_time(start_frame, video.fps)
        end_time = calc_elapsed_time(end_frame, video.fps)
        clip = video.subclip(start_time, end_time)
        clipsArray.append(clip)
        times.append([start_time, end_time])
    final = concatenate_videoclips(clipsArray)
    final.write_videofile(
        dst,
        fps=video.fps,
        codec=config["video_codec"],
        audio_codec=config["audio_codec"],
    )
    video.close()


def process(src_record_path, chara_crop_dir):
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

            clip_movie(movie_raw_file_path, clips, dst, src_record_path)
    except Exception as e:
        print("process ERROR: ", src_record_path)
        print(e)
        print(traceback.format_exc())


def prepare(charactor_name, series_name):
    chara_crop_dir = os.path.join(crop_dir, charactor_name, series_name)
    if not os.path.exists(chara_crop_dir):
        os.makedirs(chara_crop_dir, exist_ok=True)

    return chara_crop_dir


def main():
    for series_name in config["folders"]:
        charactor_dirs = fs.list_dirs(record_dir)
        pbar = tqdm(charactor_dirs)
        for charactor_name in pbar:
            print("CHARACTOR -> ", charactor_name)
            try:
                if len(config["rejected_charactors"]) > 0 and charactor_name in config["rejected_charactors"]:
                    pbar.update(1)
                    continue
                if len(config["available_charactors"]) > 0 and charactor_name not in config["available_charactors"]:
                    pbar.update(1)
                    continue
                chara_crop_dir = prepare(charactor_name, series_name)
                chara_record_dir = os.path.join(record_dir, charactor_name, series_name)
                record_pathes = fs.list_entries(chara_record_dir)
                joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
                    src_record_path=record_path,
                    chara_crop_dir=chara_crop_dir,
                ) for record_path in record_pathes])
            except Exception as e:
                print("main ERROR: ")
                print(e)
                print(traceback.format_exc())


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
