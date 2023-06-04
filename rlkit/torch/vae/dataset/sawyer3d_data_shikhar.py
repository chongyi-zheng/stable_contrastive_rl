import os.path as osp
import numpy as np
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
import time
import cv2


def get_data(N = 10000, test_p = 0.9):
    dir = '/home/shikharbahl/ros_ws/src/sawyer_control/src/sawyer_control/images_data_1/'
    images = []
    for i in range(N):
        im = cv2.imread(dir + str(i) + '.png', cv2.IMREAD_UNCHANGED)
        images.append((im / 255.0).transpose().flatten())
    images = np.array(images)
    data = np.split(images, [int(N*test_p)])
    train_data, test_data = data[0], data[1]
    return train_data, test_data

if __name__ == "__main__":
    get_data()
