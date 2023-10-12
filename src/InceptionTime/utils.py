
import numpy as np
import pandas as pd
import random

import os
#import operator

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder


def check_if_file_exits(file_name):
    return os.path.exists(file_name)


def readucr(filename, delimiter=''):
    try:
        data = np.loadtxt(filename, delimiter=None)
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y
    except:
        return [], []


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def calculate_metrics(y_true, y_pred, duration):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def transform_labels(y_train, y_val, y_test):
    """
    Transform label to min equal zero and continuous
    For example if we have [1,3,4] --->  [0,1,2]
    """
    # no validation split
    # init the encoder
    encoder = LabelEncoder()
    # concat train and test to fit
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)
    # fit the encoder
    encoder.fit(y_all)
    # transform to min zero and continuous labels
    new_y_train_test = encoder.transform(y_all)
    # resplit the train and test
    n_train = len(y_train)
    n_val = len(y_val)
    n_test = len(y_test)
    new_y_train = new_y_train_test[0:n_train]
    new_y_val = new_y_train_test[n_train:n_train+n_val]
    new_y_test = new_y_train_test[n_train+n_val:]
    return new_y_train, new_y_val, new_y_test



def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()




# def generate_array_of_colors(n):
#     # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
#     ret = []
#     r = int(random.random() * 256)
#     g = int(random.random() * 256)
#     b = int(random.random() * 256)
#     alpha = 1.0
#     step = 256 / n
#     for i in range(n):
#         r += step
#         g += step
#         b += step
#         r = int(r) % 256
#         g = int(g) % 256
#         b = int(b) % 256
#         ret.append((r / 255, g / 255, b / 255, alpha))
#     return ret




