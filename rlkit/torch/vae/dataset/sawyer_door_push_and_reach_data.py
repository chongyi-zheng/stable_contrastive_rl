import time

import numpy as np
import os.path as osp
import pickle

from gym.spaces import Box

from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorPushOpenEnv, SawyerDoorPushOpenEnv, SawyerDoorPushOpenAndReachEnv
from multiworld.core.image_env import ImageEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.images.camera import sawyer_door_env_camera, sawyer_door_env_camera
import cv2

from rlkit.util.io import local_path_from_s3_or_local_path, sync_down
from rlkit.policies.simple import RandomPolicy
from rlkit.torch import pytorch_util as ptu


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=False,
        dataset_path=None, env_class=None, env_kwargs=None, init_camera=sawyer_door_env_camera,
):
    filename = "/tmp/sawyer_door_push_open_and_reach" + str(N) + ".npy"
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        env = env_class(**env_kwargs)
        env =  ImageEnv(
            env, imsize,
            transpose=True,
            init_camera=init_camera,
            normalize=True,
        )
        oracle_sampled_data = int(N/2)
        dataset = np.zeros((N, imsize * imsize * 3))
        print('Goal Space Sampling')
        for i in range(oracle_sampled_data):
            goal = env.sample_goal()
            env.set_to_goal(goal)
            img = env._get_flat_img()
            dataset[i, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
        env._wrapped_env.min_y_pos=.6
        policy = RandomPolicy(env.action_space)
        es = OUStrategy(action_space=env.action_space, theta=0)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        print('Random Sampling')
        for i in range(oracle_sampled_data, N):
            if i % 20==0:
                env.reset()
                exploration_policy.reset()
            for _ in range(10):
                action = exploration_policy.get_action()[0]
                env.wrapped_env.step(
                    action
                )
            img = env._get_flat_img()
            dataset[i, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(
        1000,
        use_cached=False,
        show=True,
        env_class=SawyerDoorPushOpenAndReachEnv,
        env_kwargs=dict(
            max_x_pos=.1,
            max_y_pos=.8,
            frame_skip=50,
        ),
    )
