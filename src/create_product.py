import os
import traceback
import joblib
import json
import time
from tqdm import tqdm
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.resize import resize
from pathlib import Path

from lib import fs

JOB_NUM = 3

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "04.product")
output_dir = os.path.join(resource_dir, "movies")
input_dir = os.path.join(cwd, "03.movie", "crop_movies")
config_file_path = os.path.join(cwd.parent, "config.json")

config_file = open(config_file_path, 'r')
config = json.load(config_file)


def concat_movies(movie_file_pathes):
    print(movie_file_pathes)
    clips = []
    for path in movie_file_pathes:
        clip = VideoFileClip(path)
        if clip.w != config["frame_width"] or clip.h != config["frame_height"]:
            clip = resize(clip, (config["frame_width"], config["frame_height"]))
        clips.append(clip)
    final_clip = concatenate_videoclips(clips)
    return final_clip


def export_video(clip, dst):
    clip.write_videofile(
        dst,
        codec=config["video_codec"],
        audio_codec=config["audio_codec"],
    )


def export_audio(clip, dst):
    clip.audio = clip.audio.set_fps(clip.audio.fps) # NOTE: https://github.com/Zulko/moviepy/issues/1247
    clip.audio.write_audiofile(dst)


def process(movie_file_pathes, chara_dir, series_name):
    try:
        if len(movie_file_pathes) == 0:
            return

        dst_video = os.path.join(chara_dir, f"{series_name}.mp4")
        if os.path.exists(dst_video):
            clip = VideoFileClip(dst_video)
        else:
            clip = concat_movies(movie_file_pathes)
        if not os.path.exists(dst_video):
            export_video(clip, dst_video)
        fs.wait_until_generated(dst_video)

        dst_audio = os.path.join(chara_dir, f"{series_name}.mp3")
        if not os.path.exists(dst_audio):
            export_audio(clip, dst_audio)
    except Exception as e:
        print("process ERROR: ", dst_video)
        print(e)
        print(traceback.format_exc())


def prepare(charactor_name):
    chara_dir = os.path.join(output_dir, charactor_name)
    if not os.path.exists(chara_dir):
        os.makedirs(chara_dir, exist_ok=True)
    return chara_dir


def main():
    charactor_dirs = fs.list_dirs(input_dir)
    pbar = tqdm(charactor_dirs)
    for charactor_name in pbar:
        print("CHARACTOR -> ", charactor_name)
        if len(config["rejected_charactors"]) > 0 and charactor_name in config["rejected_charactors"]:
            pbar.update(1)
            continue
        if len(config["available_charactors"]) > 0 and charactor_name not in config["available_charactors"]:
            pbar.update(1)
            continue
        chara_dir = prepare(charactor_name)
        joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
            movie_file_pathes=fs.list_entries(os.path.join(input_dir, charactor_name, series_name)),
            chara_dir=chara_dir,
            series_name=series_name
        ) for series_name in config["folders"]])
        # for series_name in CLIP_TARGET_FOLDER_NAMES:
        #     process(
        #         fs.list_entries(os.path.join(input_dir, charactor_name, series_name)),
        #         chara_dir,
        #         series_name
        #     )


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
