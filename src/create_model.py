#!/usr/bin/env python
# coding: utf-8


import sys
import cv2
import os
import math
import numpy as np
import traceback
import joblib
import json
import random
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process
from sklearn.model_selection import train_test_split
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.initializers import he_normal, TruncatedNormal, Constant
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from lib import fs
from lib.image import Image


IMAGE_SIZE_PX = 112
BATCH_SIZE = 12
EPOCHS = 50
LEARNING_RATE = 0.0001
TRAIN_DATA_SIZE_RATIO = 0.7
VALID_DATA_SIZE_RATIO = 0.3
TEST_DATA_SIZE_RATIO = 0.3  # NOTE: 指定の比率でVALID_DATA_SIZE_RATIOを分割する。valid:0.3, test:0.3なら全体の0.09。

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
resource_dir = os.path.join(cwd, "01.model")
input_data_dir_path = os.path.join(resource_dir, "input")
train_data_dir_path = os.path.join(resource_dir, "train")
class_mapping_file_path = os.path.join(cwd.parent, "model", "class_mapping.json")
model_file_path = os.path.join(cwd.parent, "model", "dnn_model.h5")
checkpoint_file_path = os.path.join(cwd.parent, "model", "dnn_checkpoint.h5")

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def make_classes(dir):
    dirs = [os.path.join(dir, p.name) for p in os.scandir(dir)]
    class_pathes = []
    for d in dirs:
        class_pathes += [os.path.join(d, p.name) for p in os.scandir(d)]
    ret = []
    for class_path in class_pathes:
        ret.append({
            "name": os.path.basename(os.path.normpath(class_path)),
            "path": class_path
        })
    return ret


def make_train_data(classes):
    print(classes)
    ret = {}
    for c in classes:
        ret[c["name"]] = []
        img_file_pathes = fs.list_files(c["path"])
        class_dir_path = os.path.join(train_data_dir_path, c["name"])
        os.makedirs(class_dir_path, exist_ok=True)
        for filepath in img_file_pathes:
            image = Image()
            image.load_image_from_filepath(filepath)
            image.transform_image_for_predict_with(IMAGE_SIZE_PX)
            dst_filepath = os.path.join(class_dir_path, fs.get_filename(filepath))
            image.write_to(dst_filepath)
            print(dst_filepath)
            ret[c["name"]].append(dst_filepath)
    return ret


def preprocess(classes, train_data):
    ml_data = []
    ml_label = []
    ml_data_min_num = 1001001  # NOTE: データの偏りをなくすため少ない方に合わせる。

    for label, img_file_pathes in train_data.items():
        ml_data_min_num = min(ml_data_min_num, len(img_file_pathes))
    print('min:', ml_data_min_num)

    class_mapping = {}
    for label_index, (label, img_file_pathes) in enumerate(train_data.items()):
        sample = random.sample(img_file_pathes, ml_data_min_num)
        class_mapping[label_index] = label
        for img_file_path in img_file_pathes:
            img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            ml_data.append(img)
            ml_label.append(label_index)
    print(len(ml_data))
    fs.write_json(class_mapping_file_path, class_mapping)

    return ml_data, ml_label


def create_model(class_num):
    model = Sequential()
    model.add(Conv2D(math.floor(IMAGE_SIZE_PX / 4), kernel_size=(3, 3), input_shape=(IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(math.floor(IMAGE_SIZE_PX / 2), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(math.floor(IMAGE_SIZE_PX / 2), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(math.floor(IMAGE_SIZE_PX), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(math.floor(IMAGE_SIZE_PX / 2), activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    return model


def step_decay(epoch):
    lr = 0.001
    if epoch >= 25:
        lr /= 5
    if epoch >= 50:
        lr /= 5
    if epoch >= 100:
        lr /= 2
    return lr


def setup(class_num, ml_data, ml_label):
    data_train, _data_valid, label_train, _label_valid = train_test_split(ml_data, ml_label, test_size=VALID_DATA_SIZE_RATIO, train_size=TRAIN_DATA_SIZE_RATIO)
    data_valid, data_test, label_valid, label_test = train_test_split(_data_valid, _label_valid, test_size=TEST_DATA_SIZE_RATIO)

    na_data_train = np.array(data_train)
    na_data_valid = np.array(data_valid)
    na_data_test = np.array(data_test)

    na_label_train = np.array(label_train)
    na_label_valid = np.array(label_valid)
    na_label_test = np.array(label_test)

    print(na_data_train.shape, na_data_train[0].shape)
    na_data_train = na_data_train.reshape(na_data_train.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)  # 2次元配列を1次元に変換
    na_data_valid = na_data_valid.reshape(na_data_valid.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)
    na_data_test = na_data_test.reshape(na_data_test.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)

    # int型をfloat32型に変換
    na_data_train = na_data_train.astype('float32')
    na_data_valid = na_data_valid.astype('float32')
    na_data_test = na_data_test.astype('float32')

    # [0-255]の値を[0.0-1.0]に変換
    na_data_train /= 255
    na_data_valid /= 255
    na_data_test /= 255

    print(na_data_train.shape[0], 'train samples')
    print(na_data_valid.shape[0], 'valid samples')
    print(na_data_test.shape[0], 'test samples')

    label_train_classes = keras.utils.to_categorical(na_label_train, class_num)
    label_valid_classes = keras.utils.to_categorical(na_label_valid, class_num)
    label_test_classes = keras.utils.to_categorical(na_label_test, class_num)

    return (
        na_data_train, na_data_valid, na_data_test,
        label_train_classes, label_valid_classes, label_test_classes
    )


def train(class_num, na_data_train, na_data_valid, label_train_classes, label_valid_classes):

    model = create_model(class_num)
    model.summary()

    lr_decay = LearningRateScheduler(step_decay)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=2
    )

    modelCheckpoint = ModelCheckpoint(filepath=checkpoint_file_path,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='min',
                                      period=1)

    # adam = Adam(lr=0.0001)
    sdg = keras.optimizers.SGD(lr=LEARNING_RATE, momentum=0.9, decay=1e-4, nesterov=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sdg,
                  metrics=['accuracy'])
    history = model.fit(na_data_train, label_train_classes,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_data=(na_data_valid, label_valid_classes),
                        callbacks=[lr_decay, modelCheckpoint])

    return history, model


def evaluate(model, na_data_test, label_test_classes):
    score = model.evaluate(na_data_test, label_test_classes, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy:', score[1])
    return score


def main():
    classes = make_classes(input_data_dir_path)
    train_data = make_train_data(classes)
    ml_data, ml_label = preprocess(classes, train_data)
    class_num = len(classes)

    (
        na_data_train, na_data_valid, na_data_test,
        label_train_classes, label_valid_classes, label_test_classes
    ) = setup(class_num, ml_data, ml_label)
    history, model = train(class_num, na_data_train, na_data_valid, label_train_classes, label_valid_classes)
    loss, acc = evaluate(model, na_data_test, label_test_classes)
    model.save(model_file_path, include_optimizer=False)


if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')