import time

import numpy as np
import os.path as osp

from rlkit.envs.mujoco.sawyer_push_env import SawyerPushXYVariableEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.images.camera import (
    sawyer_init_camera_zoomed_in,
    sawyer_init_camera,
    sawyer_init_camera_zoomed_in_fixed,
    sawyer_init_camera_zoomed_out_fixed,
)
import cv2

from rlkit.util.io import local_path_from_s3_or_local_path


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=False,
        init_camera=sawyer_init_camera_zoomed_in,
        dataset_path=None,
        env_kwargs=None,
):
    if env_kwargs is None:
        env_kwargs = {}
    filename = "/tmp/sawyer_push_variable{}_{}.npy".format(
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
        env = SawyerPushXYVariableEnv(hide_goal=True, **env_kwargs)
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=init_camera,
            normalize=True,
        )
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * 3))
        for i in range(N):
            goal = env.sample_goal_for_rollout()
            hand_pos = env.sample_hand_xy()
            env.set_to_goal(goal, reset_hand=False)
            env.set_hand_xy(hand_pos)
            # img = env.reset()
            img = env.step(env.action_space.sample())[0]
            dataset[i, :] = img
            if show:
                img = img.reshape(3, 84, 84).transpose()
                img = img[::-1, :, ::-1]
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # radius = input('waiting...')
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(
        1000,
        use_cached=False,
        show=True,
        # init_camera=sawyer_init_camera_zoomed_in,
        # init_camera=sawyer_init_camera,
        init_camera=sawyer_init_camera_zoomed_in_fixed,
        # init_camera=sawyer_init_camera_zoomed_out_fixed,
    )
