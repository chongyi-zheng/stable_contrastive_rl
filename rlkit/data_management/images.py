import numpy as np


class _ImageNumpyArr:
    """
    Wrapper for a numpy array. This code automatically normalizes/unormalizes
    the image internally. This process should be completely hidden from the
    user of this class.

    im_arr = image_numpy_wrapper.zeros(10)
    # img is normalized. ImageNumpyArr automatically stores as np.uint8
    im_arr[2] = img
    # ImageNumpyArr automatically normalizes the np.uint8 and returns as np.float
    img_2 = im_arr[2]
    """

    def __init__(self, np_array):
        assert np_array.dtype == np.uint8
        self.np_array = np_array
        self.shape = self.np_array.shape
        self.size = self.np_array.size
        self.dtype = np.uint8

    def __getitem__(self, idxs):
        return normalize_image(self.np_array[idxs], dtype=np.float32)

    def __setitem__(self, idxs, value):
        if value.dtype != np.uint8:
            self.np_array[idxs] = unnormalize_image(value)
        else:
            self.np_array[idxs] = value


def zeros(shape, *args, **kwargs):
    arr = np.zeros(shape, dtype=np.uint8)
    return _ImageNumpyArr(arr)


def ones(shape, *args, **kwargs):
    arr = np.ones(shape, dtype=np.uint8)
    return _ImageNumpyArr(arr)


def from_np(np_arr):
    return _ImageNumpyArr(np_arr)


def normalize_image(image, dtype=np.float64):
    assert image.dtype == np.uint8
    return dtype(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
