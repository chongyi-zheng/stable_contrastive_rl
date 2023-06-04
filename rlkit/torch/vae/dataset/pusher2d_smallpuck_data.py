import os.path as osp
import numpy as np
from rlkit.envs.multitask.pusher2d import FullPusher2DEnv
from rlkit.envs.wrappers import ImageMujocoEnv
import time
import cv2

def get_data(N = 10000, test_p = 0.9, use_cached=True, render=False):
    filename = "/tmp/pusher2d_smallpuck_" + str(N) + ".npy"
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename).astype(np.float32)
        print("loaded data from saved file", filename)
    else:
        # if not cached
        now = time.time()
        e = FullPusher2DEnv()
        e = ImageMujocoEnv(e, 84, camera_name="topview", transpose=True, normalize=True)
        dataset = np.zeros((N, 3*84*84))
        for i in range(N):
            if i % 100 == 0:
                e.reset()
            u = np.random.rand(3) * 4 - 2
            img, _, _, _ = e.step(u)
            dataset[i, :] = img
            if render:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
        print("done making training data", filename, time.time() - now, "mean", dataset.mean())
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset

if __name__ == "__main__":
    get_data(10000, use_cached=False, render=False)
