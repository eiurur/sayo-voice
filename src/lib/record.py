import os
import traceback
import numpy as np
import copy

from . import fs
from .stopwatch import StopWatch


class Record:
    def __init__(self):
        self.periods = []
        self.start_frame = -1

    @property
    def dir_format(self):
        pass

    @property
    def label(self):
        pass

    @property
    def model(self):
        pass

    @property
    def threshold(self):
        pass

    @property
    def skip_frame_interval(self):
        pass

    @property
    def px(self):
        pass

    @dir_format.setter
    def dir_format(self, dir_format):
        self.__dir_format = dir_format

    @label.setter
    def label(self, label):
        self.__label = label

    @model.setter
    def model(self, model):
        self.__model = model

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    @skip_frame_interval.setter
    def skip_frame_interval(self, skip_frame_interval):
        self.__skip_frame_interval = skip_frame_interval

    @px.setter
    def px(self, px):
        self.__px = px

    @dir_format.getter
    def dir_format(self):
        return self.__dir_format

    @label.getter
    def label(self):
        return self.__label

    def get_label_index(self):
        index, name = self.__label
        return index

    def get_label_name(self):
        index, name = self.__label
        return name

    def __predict(self, im):
        img_predict = [im]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], self.__px, self.__px, 1)
        data_predict = data_predict.astype('float32')
        data_predict /= 255

        predictions = self.__model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        return pred_class, pred_proba

    def prepare(self):
        dir_path = self.__dir_format.format(self.get_label_name())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def predict(self, im):
        try:
            _im = copy.deepcopy(im)
            _im.transform_image_for_predict_with(self.__px)
            pred_class, pred_proba = self.__predict(_im.get_image())
            return pred_class, pred_proba
        except Exception as e:
            return None, None

    def record(self, frame_idx, im, pred_class, pred_proba):
        if self.start_frame == -1 and (pred_proba >= self.__threshold and str(self.get_label_index()) == str(pred_class)):
            self.start_frame = frame_idx
            print(self.get_label_name(), pred_class, pred_proba)
            # self.debug_image(im, f"{frame_idx}_{pred_class}_{pred_proba}.jpg")
        elif self.start_frame != -1 and (pred_proba < self.__threshold or str(self.get_label_index()) != str(pred_class)):
            end_frame = frame_idx - self.__skip_frame_interval
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            print(self.get_label_name(), pred_class, pred_proba)
            print(self.periods)

    """
    def compare(self, frame_idx, crop_im):
        try:
            crop_im.transform_image_for_predict_with(self.__px)
            pred_class, pred_proba = self.predict(crop_im.get_image())
        except Exception as e:
            print(traceback.format_exc())
            return False

        if self.start_frame == -1 and (pred_proba >= self.__threshold and str(self.get_label_index()) == str(pred_class)):
            self.start_frame = frame_idx
            print(pred_class, pred_proba)
            self.debug_image(crop_im, f"{frame_idx}_{pred_class}_{pred_proba}.jpg")
        elif self.start_frame != -1 and (pred_proba < self.__threshold or str(self.get_label_index()) != str(pred_class)):
            end_frame = frame_idx
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            print(pred_class, pred_proba)
            print(self.get_label_name(), self.periods)
            self.debug_image(crop_im, f"{frame_idx}_{pred_class}_{pred_proba}.jpg")
    """

    def debug_image(self, im, filename):
        dir_path = self.__dir_format.format(self.get_label_name())
        c_dir = os.path.join(dir_path, "crop")
        os.makedirs(c_dir, exist_ok=True)
        im.write_to(os.path.join(c_dir, filename))  # DEBUG

    def write_to_file(self, prefix_data, movie_file_name_without_ext):
        if prefix_data is None:
            prefix_data = []
        dir_path = self.__dir_format.format(self.get_label_name())
        record_file = os.path.join(dir_path, "{}.txt".format(movie_file_name_without_ext))
        if os.path.exists(record_file):
            return

        data = prefix_data + self.periods
        with open(record_file, "w", encoding='UTF-8') as f:
            f.write('\n'.join(data))
