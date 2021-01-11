import math
import os

from . import fs


class Record:
    def __init__(self):
        self.periods = []
        self.start_frame = -1

    @property
    def dir_format(self):
        return self.__dir_format

    @property
    def label(self):
        return self.__label

    @property
    def threshold(self):
        pass

    @property
    def skip_frame_interval(self):
        pass

    @dir_format.setter
    def dir_format(self, dir_format):
        self.__dir_format = dir_format

    @label.setter
    def label(self, label):
        self.__label = label

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @skip_frame_interval.setter
    def skip_frame_interval(self, skip_frame_interval):
        self.__skip_frame_interval = skip_frame_interval

    def get_label_index(self):
        index, name = self.__label
        return index

    def get_label_name(self):
        index, name = self.__label
        return name

    def get_config_data(self):
        return {
            "threshold": str(self.__threshold),
            "skip_frame_interval": str(self.__skip_frame_interval),
        }

    def get_dir_path(self):
        return self.dir_format.format(self.get_label_name())

    def prepare(self):
        dir_path = self.__dir_format.format(self.get_label_name())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def record(self, frame_idx, im, pred_class, pred_proba):
        if self.start_frame == -1 and (pred_proba >= self.__threshold and str(self.get_label_index()) == str(pred_class)):
            self.start_frame = frame_idx
            print("record:start ", self.get_label_name(), pred_class, pred_proba)
            # self.debug_image(im, f"start_{frame_idx}_{pred_class}_{pred_proba}.jpg")
        elif self.start_frame != -1 and (pred_proba < self.__threshold or str(self.get_label_index()) != str(pred_class)):
            end_frame = frame_idx - math.ceil(self.__skip_frame_interval / 2)
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            print("record:end   ", self.get_label_name(), pred_class, pred_proba)
            # self.debug_image(im, f"end_{frame_idx}_{pred_class}_{pred_proba}.jpg")

    def debug_image(self, im, filename):
        dir_path = self.__dir_format.format(self.get_label_name())
        c_dir = os.path.join(dir_path, "crop")
        os.makedirs(c_dir, exist_ok=True)
        im.write_to(os.path.join(c_dir, filename))

    def write_to_file(self, record_info, filename_without_ext):
        if record_info is None:
            record_info = {}

        dir_path = self.__dir_format.format(self.get_label_name())
        record_file = os.path.join(dir_path, "{}.json".format(filename_without_ext))
        if os.path.exists(record_file):
            return
        record_info["periods"] = self.periods
        fs.write_json(record_file, record_info)
