import os
import cv2
import traceback
import hashlib
from tqdm import tqdm

from . import fs
from .image import Image


class Movie:
    def __init__(self, movie_path, skip, records, predictor):
        self.movie_path = movie_path
        self.skip = skip or 3
        self.records = records
        self.predictor = predictor

    def is_completed_clip(self):
        bool = False
        for record in self.records:
            dir_path = record.get_dir_path()
            movie_file_name_without_ext = fs.get_filename_without_ext(self.movie_path)
            record_file = os.path.join(dir_path, "{}.json".format(movie_file_name_without_ext))
            if os.path.exists(record_file):
                bool = True
                break
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

    def capture(self):
        cap = cv2.VideoCapture(self.movie_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print("PATH: ", self.movie_path)
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
            try:
                image = Image()
                image.set_image(crop)
                pred_class, pred_proba = self.predictor.predict(image)
                for record in self.records:
                    record.record(frame_idx, image, pred_class, pred_proba)
            except Exception as e:
                print(traceback.format_exc())
                pbar.update(1)
        cap.release()

    def write_period_to_file(self):
        [filename, ext] = fs.get_filename_and_ext(self.movie_path)
        hash_code = hashlib.md5(filename.encode()).hexdigest()
        hashed_filename = "{}{}".format(hash_code, ext)
        for record in self.records:
            config_data = record.get_config_data()
            record_info = {
                "movie_path": self.movie_path,
                "hashed_filename": hashed_filename,
                "config_data": config_data
            }
            record.write_to_file(record_info, filename)
