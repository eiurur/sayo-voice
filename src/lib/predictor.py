import copy
import numpy as np


class Predictor:
    def __init__(self, model, px):
        self.model = model
        self.px = px

    @property
    def model(self):
        pass

    @model.setter
    def model(self, model):
        self.__model = model

    def __predict(self, im):
        img_predict = [im]
        data_predict = np.asarray(img_predict)
        data_predict = data_predict.reshape(data_predict.shape[0], self.px, self.px, 1)
        data_predict = data_predict.astype('float32')
        data_predict /= 255

        predictions = self.__model.predict(data_predict)
        pred_class = predictions.argmax()
        pred_proba = predictions[0][pred_class] * 100
        return pred_class, pred_proba

    def predict(self, im):
        try:
            _im = copy.deepcopy(im)
            _im.transform_image_for_predict_with(self.px)
            pred_class, pred_proba = self.__predict(_im.get_image())
            return pred_class, pred_proba
        except Exception as e:
            return None, None
