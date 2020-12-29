
import os
import traceback
import numpy as np
import fs

class Record:
    def __init__(self):
        self.periods = []
        self.start_frame = -1

    @property
    def dir_format(self):
        pass

    @dir_format.setter
    def dir_format(self, dir_format):
        self.__dir_format = dir_format

    @property
    def label(self):
        pass

    @label.setter
    def label(self, label):
        self.__label_i, self.__label_name = label

    @property
    def model(self):
        pass

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def threshold(self):
        pass

    @threshold.setter
    def threshold(self, threshold):
        self.__threshold = threshold

    def prepare(self):
        dir_path = self.__dir_format.format(self.__label_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def predict(self, im, px):
        img_predict = [im]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], px, px, 1) # 2次元配列を1次元に変換
        data_predict = data_predict.astype('float32')   # int型をfloat32型に変換
        data_predict /= 255                        # [0-255]の値を[0.0-1.0]に変換

        predictions = self.__model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        return pred_class, pred_proba

    def compare(self, frame_idx, crop_im, px):
        try:
            crop_im.transform_image_for_predict_with(px)
            pred_class, pred_proba = self.predict(crop_im.get_image(), px)
        except Exception as e:
            print(traceback.format_exc())
            return False

        print(pred_class, pred_proba)
        if self.start_frame == -1 and (pred_proba >= self.__threshold and str(self.__label_i) == str(pred_class)):
            self.start_frame = frame_idx
        if self.start_frame != -1 and (pred_proba < self.__threshold or str(self.__label_i) != str(pred_class)):
            end_frame = frame_idx
            self.periods.append("{}-{}".format(self.start_frame, end_frame))
            self.start_frame = -1
            print(pred_class, pred_proba)
            print(self.__label_name, self.periods)

    def write(self, prefix_data, movie_file_name_without_ext):
        if prefix_data is None:
            prefix_data = []
        dir_path = self.__dir_format.format(self.__label_name)
        record_file = os.path.join(dir_path, "{}.txt".format(movie_file_name_without_ext))
        if os.path.exists(record_file):
            return

        data = prefix_data + self.periods
        with open(record_file, "w", encoding='UTF-8') as f:
            f.write('\n'.join(data))
