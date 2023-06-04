import time

import numpy as np
import os.path as osp

from rlkit.envs.mujoco.sawyer_push_env import SawyerPushXYEnv, \
    SawyerPushXYEasyEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.images.camera import sawyer_init_camera, \
    sawyer_init_camera_zoomed_in
import cv2

from rlkit.util.io import local_path_from_s3_or_local_path


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=False,
        init_camera=sawyer_init_camera_zoomed_in,
        dataset_path=None,
):
    filename = "/tmp/sawyer_push_new_easy{}_{}.npy".format(
        str(N),
        init_camera.__name__,
    )
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
        N = dataset.shape[0]
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = SawyerPushXYEasyEnv(hide_goal=True)
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=init_camera,
            normalize=True,
        )
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * 3))
        for i in range(N):
            env.reset()
            for _ in range(100):
                action = env.wrapped_env.action_space.sample()
                # action[0] = 0
                # action[1] = 1
                env.wrapped_env.step(action)
            img = env.step(env.action_space.sample())[0]
            dataset[i, :] = img
            print(i)
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(
        10000,
        use_cached=False,
        init_camera=sawyer_init_camera_zoomed_in,
    )
