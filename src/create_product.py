import os
import traceback
import joblib
from tqdm import tqdm
from datetime import datetime
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.fx.resize import resize
from pathlib import Path

from lib import fs


CLIP_TARGET_FOLDER_NAMES = ["s1", "s2", "band"]
REJECTED_CHARACTOR_NAMES = ["background", "moyo", "logo"]
AVAILABLE_CHARACTOR_NAMES = ["sayo", "hina", "tsugumi", "kokoro"]
JOB_NUM = len(CLIP_TARGET_FOLDER_NAMES)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
VIDEO_CODEC = "h264_nvenc"  # GPU
AUDIO_CODEC = "aac"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "04.product")
output_dir = os.path.join(resource_dir, "movies")
input_dir = os.path.join(cwd, "03.movie", "crop_movies")


def concat_movies(movie_file_pathes, dst):
    print(movie_file_pathes, dst)
    clips = []
    for path in movie_file_pathes:
        clip = VideoFileClip(path)
        print(clip.w, clip.h, clip.w != FRAME_WIDTH)
        if clip.w != FRAME_WIDTH or clip.h != FRAME_HEIGHT:
            clip = resize(clip, (FRAME_WIDTH, FRAME_HEIGHT))
        clips.append(clip)
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(
        dst,
        codec=VIDEO_CODEC,
        audio_codec=AUDIO_CODEC,
    )


def process(movie_file_pathes, chara_dir, series_name):
    try:
        if len(movie_file_pathes) == 0:
            return
        dst = os.path.join(chara_dir, f"{series_name}.mp4")
        if os.path.exists(dst):
            return
        concat_movies(movie_file_pathes, dst)
    except Exception as e:
        print("process ERROR: ", dst)
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
        if len(REJECTED_CHARACTOR_NAMES) > 0 and charactor_name in REJECTED_CHARACTOR_NAMES:
            pbar.update(1)
            continue
        if len(AVAILABLE_CHARACTOR_NAMES) > 0 and charactor_name not in AVAILABLE_CHARACTOR_NAMES:
            pbar.update(1)
            continue
        chara_dir = prepare(charactor_name)
        joblib.Parallel(n_jobs=JOB_NUM)([joblib.delayed(process)(
            movie_file_pathes=fs.list_entries(os.path.join(input_dir, charactor_name, series_name)),
            chara_dir=chara_dir,
            series_name=series_name
        ) for series_name in CLIP_TARGET_FOLDER_NAMES])
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
