import numpy as np


def preprocess_image(data):
    _shape = list(data.shape[:-3])
    data = np.reshape(data, [-1, 3, 48, 48])
    data = np.transpose(data, [0, 1, 3, 2])
    data = np.reshape(data, _shape + [3, 48, 48])
    data = data - 0.5
    return data
