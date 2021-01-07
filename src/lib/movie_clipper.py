
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

    def is_completed_clip(self, cache_filepath):
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
            return rgb[775:850, 220:615]
        if height == 720 and width == 1280:
            resized = cv2.resize(rgb, (1920, 1080))
            return resized[775:850, 220:615]
        return None

    def clip_frame(self, crop):
        crop_image_path = os.path.join(self.__output_dir, fs.add_prefix_to('{}.png'.format(self.pid)))
        os.makedirs(os.path.dirname(crop_image_path), exist_ok=True)
        try:
            cv2.imwrite(crop_image_path, crop)
        except Exception as e:
            print("imwrite ERROR: ", crop_image_path)
            return False

    def capture(self, pid):
        cap = cv2.VideoCapture(self.src_movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.src_movie_path)
        print("FRAME_COUNT: ", frame_count)
        print("FPS: ", fps)

        start_pos = 0
        pbar = tqdm(range(start_pos, frame_count, int(fps / fps)))
        for frame_idx in pbar:
            if frame_idx % self.skip != 0:
                pbar.update(1)
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            _, frame = cap.read()
            if frame is None:
                pbar.update(1)
                continue

            crop = self.crop_name_area(frame)
            if crop is None:
                pbar.update(1)
                break

            self.clip_frame(crop)
        cap.release()
