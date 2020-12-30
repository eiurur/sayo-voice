import cv2
import hashlib
import os
from tqdm import tqdm

from . import fs
from .image import Image


class Movie:
    def __init__(self, src_movie_path, skip, records):
        self.src_movie_path = src_movie_path
        self.skip = skip or 3
        self.records = records

    def isCompltedClip(self):
        bool = False
        for record in self.records:
            dir_path = record.dir_format.format(record.get_label_name())
            movie_file_name_without_ext = fs.get_filename_without_ext(self.src_movie_path)
            record_file = os.path.join(dir_path, "{}.txt".format(movie_file_name_without_ext))
            if os.path.exists(record_file):
                bool = True
                break
        return bool

    def crop_name_area(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = rgb.shape[:2]
        if height == 1080 and width == 1920:
            return rgb[780:850, 230:450]
        if height == 720 and width == 1280:
            return rgb[0:0, 0:0]
        return None

    def capture(self):
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

            image = Image()
            image.set_image(crop)
            for record in self.records:
                record.compare(frame_idx, image)
        cap.release()

    def write_period_to_file(self):
        for record in self.records:
            [filename, ext] = fs.get_filename_and_ext(self.src_movie_path)
            hs = hashlib.md5(filename.encode()).hexdigest()
            ascii_filename = "{}{}".format(hs, ext)
            prefix_data = [self.src_movie_path, ascii_filename]
            movie_file_name_without_ext = fs.get_filename_without_ext(self.src_movie_path)
            record.write(prefix_data, movie_file_name_without_ext)
