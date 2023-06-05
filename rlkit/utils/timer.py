"""TensorFlow Task Generators API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


class Timer(object):

    def __init__(self, keys):
        assert isinstance(keys, list)
        self._keys = keys
        self._keys = keys

        self.reset()

    def reset(self):
        self._start_time = {
            key: None for key in self._keys
        }
        self._time_acc = {
            key: 0.0 for key in self._keys
        }
        self._count = {
            key: 0 for key in self._keys
        }

    @property
    def time_acc(self):
        return self._time_acc

    def tic(self, key):
        assert key in self._keys
        self._start_time[key] = time.time()

    def toc(self, key):
        assert self._start_time[key] is not None
        self._time_acc[key] += time.time() - self._start_time[key]
        self._count[key] += 1
        self._start_time[key] = None

    def accumulated_time(self, key):
        return self._time_acc[key]

    def average_time(self, key):
        assert self._count[key] > 0
        return self._time_acc[key] / self._count[key]
