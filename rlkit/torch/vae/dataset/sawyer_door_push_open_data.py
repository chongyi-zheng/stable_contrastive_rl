import time

import numpy as np
import os.path as osp
import pickle

from gym.spaces import Box

from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorPushOpenEnv, SawyerDoorPushOpenEnv
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
        dataset_path=None, policy_path=None, ratio_oracle_policy_data_to_random=1/2, action_space_sampling=False, env_class=None, env_kwargs=None,
        action_plus_random_sampling=False, init_camera=sawyer_door_env_camera,
):
    if policy_path is not None:
        filename = "/tmp/sawyer_door_push_open_oracle+random_policy_data_closer_zoom_action_limited" + str(N) + ".npy"
    elif action_space_sampling:
        filename = "/tmp/sawyer_door_push_open_zoomed_in_action_space_sampling" + str(N) + ".npy"
    else:
        filename = "/tmp/sawyer_door_push_open" + str(N) + ".npy"
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    elif action_space_sampling:
        env = SawyerDoorPushOpenEnv(**env_kwargs)
        env = ImageEnv(
            env, imsize,
            transpose=False,
            init_camera=sawyer_door_env_camera,
            normalize=False,
        )
        action_space = Box(
            np.array([-env.max_x_pos, .5, .06]),
            np.array([env.max_x_pos, env.max_y_pos, .06])
        )
        dataset = np.zeros((N, imsize * imsize * 3))
        for i in range(N):
            env.set_to_goal_pos(action_space.sample()) #move arm to spot
            goal = env.sample_goal()
            env.set_to_goal(goal)
            img = env.get_image().flatten()
            dataset[i, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
        info['env'] = env
    elif action_plus_random_sampling:
        env = env_class(**env_kwargs)
        env =  ImageEnv(
            env, imsize,
            transpose=True,
            init_camera=init_camera,
            normalize=True,
        )
        action_space = Box(
            np.array([-env.max_x_pos, .5, .06]),
            np.array([env.max_x_pos, .6, .06])
        )
        action_sampled_data = int(N/2)
        dataset = np.zeros((N, imsize * imsize * 3))
        print('Action Space Sampling')
        for i in range(action_sampled_data):
            env.set_to_goal_pos(action_space.sample())  # move arm to spot
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
        for i in range(action_sampled_data, N):
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
        env._wrapped_env.min_y_pos = .5
        info['env'] = env
    else:
        now = time.time()
        env = SawyerDoorPushOpenEnv(max_angle=.5)
        env = ImageEnv(
            env, imsize,
            transpose=True,
            init_camera=sawyer_door_env_camera,
            normalize=True,
        )
        info['env'] = env
        policy = RandomPolicy(env.action_space)
        es = OUStrategy(action_space=env.action_space, theta=0)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        dataset = np.zeros((N, imsize * imsize * 3))
        for i in range(N):
            if i % 100==0:
                env.reset()
                exploration_policy.reset()
            for _ in range(25):
                # env.wrapped_env.step(
                #     env.wrapped_env.action_space.sample()
                # )
                action = exploration_policy.get_action()[0]
                env.wrapped_env.step(
                    action
                )
            goal = env.sample_goal_for_rollout()
            env.set_to_goal(goal)
            img = env.step(env.action_space.sample())[0]
            dataset[i, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
        print("done making training data", filename, time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(
        100,
        use_cached=False,
        show=True,
        env_class=SawyerDoorPushOpenEnv,
        env_kwargs=dict(
            max_x_pos=.1,
            max_y_pos=.8,
            frame_skip=50,
        ),
        action_plus_random_sampling=True,
    )
