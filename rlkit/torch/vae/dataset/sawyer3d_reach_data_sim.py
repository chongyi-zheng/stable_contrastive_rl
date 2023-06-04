import time

import numpy as np
import os.path as osp

from gym.spaces import Box

from multiworld.core.image_env import unormalize_image, ImageEnv
from rlkit.envs.mujoco.sawyer_reach_env import SawyerReachXYZEnv
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.images.camera import sawyer_xyz_reacher_camera, sawyer_init_camera, \
    sawyer_xyz_camera, sawyer_torque_reacher_camera
import cv2

from rlkit.util.io import local_path_from_s3_or_local_path
from rlkit.policies.simple import RandomPolicy


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=False,
        dataset_path=None, action_space_sampling=False, init_camera=None, env_class=None, env_kwargs=None,
):
    filename = "/tmp/sawyer_xyz_pos_control_new_zoom_cam" + str(N) +'.npy'
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    else:
        now = time.time()
        if env_kwargs == None:
            env_kwargs = dict()
        env = env_class(**env_kwargs)
        env = ImageEnv(
            env, imsize,
            transpose=True,
            init_camera=init_camera,
            normalize=True,
        )
        dataset = np.zeros((N, imsize * imsize * 3), dtype=np.uint8)
        if action_space_sampling:
            action_space = Box(
                np.array([-.1, .5, 0]),
                np.array([.1, .7, .5])
            )
            for i in range(N):
                env.set_to_goal(env.sample_goal())
                img = env._get_flat_img()
                dataset[i, :] = unormalize_image(img)
                if show:
                    cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                    cv2.waitKey(1)
                print(i)
            info['env'] = env
        else:
            policy = RandomPolicy(env.action_space)
            es = OUStrategy(action_space=env.action_space, theta=0)
            exploration_policy = PolicyWrappedWithExplorationStrategy(
                exploration_strategy=es,
                policy=policy,
            )
            for i in range(N):
                # Move the goal out of the image
                env.wrapped_env.set_goal(np.array([100, 100, 100]))
                if i % 50 == 0:
                    print('Reset')
                    env.reset()
                    exploration_policy.reset()
                for _ in range(1):
                    action = exploration_policy.get_action()[0] * 10
                    env.wrapped_env.step(
                        action
                    )
                img = env.step(env.action_space.sample())[0]
                dataset[i, :] = img
                if show:
                    cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                    cv2.waitKey(1)
                print(i)

        print("done making training data", time.time() - now)
        np.save(filename, dataset)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(15000, use_cached=False, show=False, action_space_sampling=True, camera=sawyer_torque_reacher_camera)
