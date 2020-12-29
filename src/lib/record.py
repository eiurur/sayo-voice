
import os
import traceback
import numpy as np

from . import fs

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

    def prepare(self):
        dir_path = self.__dir_format.format(self.get_label_name())
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def predict(self, im):
        img_predict = [im]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], self.__px, self.__px, 1) # 2次元配列を1次元に変換
        data_predict = data_predict.astype('float32')   # int型をfloat32型に変換
        data_predict /= 255                             # [0-255]の値を[0.0-1.0]に変換

        predictions = self.__model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        return pred_class, pred_proba

    def compare(self, frame_idx, crop_im):
        try:
            crop_im.transform_image_for_predict_with(self.__px)
            pred_class, pred_proba = self.predict(crop_im.get_image())
        except Exception as e:
            print(traceback.format_exc())
            return False
            
        # crop_im.write_to("crop.jpg") # DEBUG
        # print(pred_class, pred_proba)
        if self.start_frame == -1 and (pred_proba >= self.__threshold and str(self.get_label_index()) == str(pred_class)):
            self.start_frame = frame_idx
        if self.start_frame != -1 and (pred_proba < self.__threshold or str(self.get_label_index()) != str(pred_class)):
            end_frame = frame_idx
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            # print(pred_class, pred_proba)
            # print(self.get_label_name(), self.periods)

    def write(self, prefix_data, movie_file_name_without_ext):
        if prefix_data is None:
            prefix_data = []
        dir_path = self.__dir_format.format(self.get_label_name())
        record_file = os.path.join(dir_path, "{}.txt".format(movie_file_name_without_ext))
        if os.path.exists(record_file):
            return

        data = prefix_data + self.periods
        with open(record_file, "w", encoding='UTF-8') as f:
            f.write('\n'.join(data))
