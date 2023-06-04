import time

import numpy as np
import os.path as osp

from rlkit.envs.mujoco.sawyer_gripper_env import SawyerPushXYEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.images.camera import sawyer_init_camera
import cv2


def get_data(N = 10000, test_p = 0.9, use_cached=True, imsize=84):
    filename = "/tmp/sawyer_" + str(N) + ".npy"
    info = {}
    if use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        env = SawyerPushXYEnv()
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=sawyer_init_camera,
            normalize=True,
        )
        info['env'] = env

        dataset = np.zeros((N, imsize*imsize*3))
        for i in range(N):
            env.wrapped_env.set_goal(env.wrapped_env.sample_goal_for_rollout())
            env.reset()
            for _ in range(50):
                env.wrapped_env.step(
                    env.wrapped_env.action_space.sample()
                )
            img = env.step(env.action_space.sample())[0]
            dataset[i, :] = img
            # cv2.imshow('img', img.reshape(3, 84, 84).transpose())
            # cv2.waitKey(1)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info

if __name__ == "__main__":
    get_data(10000, use_cached=False)
