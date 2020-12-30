
import os
import cv2
from tqdm import tqdm

from . import fs


class MovieClipper:
    def __init__(self, src_movie_path, skip, pid):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.pid = pid

    @property
    def output_dir(self):
        pass

    @output_dir.setter
    def output_dir(self, output_dir):
        self.__output_dir = output_dir

    def get_movie_file_name(self):
        return os.path.splitext(os.path.basename(self.src_movie_path))[0]

    def caching_to(self, cache_filepath):
        key = self.get_movie_file_name()
        with open(cache_filepath, mode='a') as f:
            f.write(f"\n{key}")

    def isCompltedClip(self, cache_filepath):
        bool = False

        if not os.path.isfile(cache_filepath):
            return bool

        with open(cache_filepath) as f:
            lines = f.readlines()
            key = self.get_movie_file_name()
            if key in lines:
                bool = True

        return bool

    def crop_name_area(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height == 1080 and width == 1920:
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            return rgb[0:0, 0:0]
        return None

    def clip_frame(self, frame_idx, crop):
        crop_image_path = os.path.join(self.__output_dir, fs.add_prefix_to('{}.png'.format(self.pid)))
        os.makedirs(os.path.dirname(crop_image_path), exist_ok=True)
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", pid, crop_image_path)
            print(traceback.format_exc())
            return False

    def capture(self, pid):
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0  # fps:  fps * (60 * 20) .. 20分

        pbar = tqdm(range(start_pos, frame_count, int(fps / fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_pos = str(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            ret, frame = cap.read()
            if frame is None:
                continue

            crop = self.crop_name_area(frame)
            if crop is None:
                break

            self.clip_frame(frame_idx, crop)
        cap.release()