from __future__ import division
from keras.layers import Input
from keras.models import Model
import os
import numpy as np
from config import *
from utilities import preprocess_images, postprocess_predictions
from models import acl_vgg
from scipy.misc import imread, imsave
from math import ceil


def get_test(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    start = 0
    while True:
        Xims = np.zeros((1, num_frames, shape_r, shape_c, 3))
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0, 0:min(len(images)-start, num_frames), :] = np.copy(X)
        yield Xims  #
        start = min(start + num_frames, len(images))
