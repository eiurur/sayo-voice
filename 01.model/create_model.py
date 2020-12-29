#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function
import sys
import cv2
import os
import math
import glob
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import traceback
import joblib
import subprocess
import json
import pprint
import random
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from time import sleep
from multiprocessing import current_process, Pool, Process
from sklearn.model_selection import train_test_split
from hashlib import md5
from time import localtime
from pathlib import Path
from numba import cuda

# TensorFlow と tf.keras のインポート
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras.initializers import he_normal, TruncatedNormal, Constant
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

cwd = Path(__file__).parent
asset_data_dir_path = os.path.join(cwd.parent, "assets")
train_data_dir_path = os.path.join(cwd.parent, "train")
class_mapping_file_path = os.path.join(cwd.parent, "class_mapping.json")

IMAGE_SIZE_PX = 112
BATCH_SIZE = 12
EPOCHS = 50
MODEL_NAME = "dnn_model.h5"


DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

# TensorFlow GPUメモリの割合を制限する
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

def resize_with_padding(img):
    h, w, c = img.shape
    longest_edge = max(h, w)
    top = 0
    bottom = 0
    left = 0
    right = 0
    if h < longest_edge:
        diff_h = longest_edge - h
        top = diff_h // 2
        bottom = diff_h - top
    elif w < longest_edge:
        diff_w = longest_edge - w
        left = diff_w // 2
        right = diff_w - left
    else:
        pass
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def list_files(dir):
    return [os.path.join(dir, p.name) for p in os.scandir(dir)]


def get_filename(p):
    return os.path.basename(p)

def get_filename_without_ext(p):
    return os.path.splitext(os.path.basename(p))[0]

def get_filename_and_ext(p):
    return os.path.splitext(os.path.basename(p))

def write_json(file_path, data):
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def load_json(file_path):
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)
        return json_data
    return None


def add_prefix_to(filename):
    prefix = md5(str(localtime()).encode('utf-8')).hexdigest()
    return f"{prefix}_{filename}"

def make_train_data(classes):
    print(classes)
    ret = {}
    for c in classes:
        ret[c["name"]] = []
        img_file_pathes = list_files(c["path"])
        class_dir_path = os.path.join(train_data_dir_path, c["name"])
        os.makedirs(class_dir_path, exist_ok=True)
        for filepath in img_file_pathes:
            img = cv2.imread(filepath)
            img = resize_with_padding(img)
            img = cv2.resize(img, dsize=(IMAGE_SIZE_PX, IMAGE_SIZE_PX))
            to = os.path.join(class_dir_path, get_filename(filepath))
            print(to)
            cv2.imwrite(to, img)
            ret[c["name"]].append(to)
    return ret

def preprocess(classes, train_data):
    ml_data = []
    ml_label = []
    ml_data_min_num = 1001001 # データの偏りをなくすため少ない方に合わせる。
    for label, img_file_pathes in train_data.items():
        ml_data_min_num = min(ml_data_min_num, len(img_file_pathes))
    print('min:', ml_data_min_num)

    class_mapping = {}
    for label_i, (label, img_file_pathes) in enumerate(train_data.items()):
        sample = random.sample(img_file_pathes, ml_data_min_num) 
        class_mapping[label_i] = label
        for i, img_file_path in enumerate(img_file_pathes):
            img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            ml_data.append(img)
            ml_label.append(label_i)
    print(len(ml_data))
    write_json(class_mapping_file_path, class_mapping)
    return ml_data, ml_label


def create_model(class_num):
    pprint.pprint(get_gpu_info())

    model = Sequential()
    model.add(Conv2D(math.floor(IMAGE_SIZE_PX/4), kernel_size=(3, 3), input_shape=(IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(math.floor(IMAGE_SIZE_PX/2), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(math.floor(IMAGE_SIZE_PX/2), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(math.floor(IMAGE_SIZE_PX), (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(math.floor(IMAGE_SIZE_PX/2), activation='relu'))
    model.add(Dense(class_num, activation='softmax'))

    return model

# 学習率
def step_decay(epoch):
    lr = 0.001
    if epoch >= 25: lr /= 5
    if epoch >= 50: lr /= 5
    if epoch >= 100: lr /= 2
    return lr

def setup(class_num, ml_data, ml_label):
    train_size = 0.7
    test_size = 0.3
    data_train, _data_test, label_train, _label_test = train_test_split(ml_data, ml_label, test_size=test_size, train_size=train_size)
    valid_size = 0.7
    data_test, data_valid, label_test, label_valid = train_test_split(_data_test, _label_test, test_size=valid_size)

    na_data_train = np.array(data_train)
    na_data_valid = np.array(data_valid)
    na_data_test = np.array(data_test)
    na_label_train = np.array(label_train)
    na_label_valid = np.array(label_valid)
    na_label_test = np.array(label_test)

    print(na_data_train.shape, na_data_train[0].shape)
    na_data_train = na_data_train.reshape(na_data_train.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1) # 2次元配列を1次元に変換
    na_data_valid = na_data_valid.reshape(na_data_valid.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)
    na_data_test = na_data_test.reshape(na_data_test.shape[0], IMAGE_SIZE_PX, IMAGE_SIZE_PX, 1)
    na_data_train = na_data_train.astype('float32')   # int型をfloat32型に変換
    na_data_valid = na_data_valid.astype('float32')
    na_data_test = na_data_test.astype('float32')
    na_data_train /= 255                        # [0-255]の値を[0.0-1.0]に変換
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

    modelCheckpoint = ModelCheckpoint(filepath = 'dnn.h5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='min',
                                      period=1)

    # adam = Adam(lr=0.0001)
    sdg = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-4, nesterov=False)
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
    print(cwd)
    classes = make_classes(asset_data_dir_path)
    train_data = make_train_data(classes)
    ml_data, ml_label = preprocess(classes, train_data)

    class_num = len(classes)
    (
        na_data_train, na_data_valid, na_data_test,
        label_train_classes, label_valid_classes, label_test_classes
    ) = setup(class_num,  ml_data, ml_label)
    history, model = train(class_num, na_data_train, na_data_valid, label_train_classes, label_valid_classes)
    loss, acc = evaluate(model, na_data_test, label_test_classes)
    model.save(MODEL_NAME, include_optimizer=False)

if __name__ == "__main__":
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    main()
    print('finished')
