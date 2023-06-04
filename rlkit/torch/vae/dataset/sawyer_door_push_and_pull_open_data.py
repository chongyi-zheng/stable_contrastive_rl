import numpy as np
import os.path as osp
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerPushAndPullDoorEnv
from multiworld.core.image_env import ImageEnv, unormalize_image
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.images.camera import sawyer_door_env_camera
import cv2
from rlkit.util.io import local_path_from_s3_or_local_path
from rlkit.policies.simple import RandomPolicy


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=False,
        dataset_path=None, policy_path=None, action_space_sampling=False, env_class=SawyerPushAndPullDoorEnv, env_kwargs=None,
        action_plus_random_sampling=False, init_camera=sawyer_door_env_camera, ratio_action_sample_to_random=1 / 2, env_id=None,
):
    if policy_path is not None:
        filename = "/tmp/sawyer_door_push_and_pull_open_oracle+random_policy_data_closer_zoom_action_limited" + str(N) + ".npy"
    elif action_space_sampling:
        filename = "/tmp/sawyer_door_push_and_pull_open_zoomed_in_action_space_sampling" + str(N) + ".npy"
    else:
        filename = "/tmp/sawyer_door_push_and_pull_open" + str(N) + ".npy"
    info = {}
    if dataset_path is not None:
        filename = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(filename)
    elif use_cached and osp.isfile(filename):
        dataset = np.load(filename)
        print("loaded data from saved file", filename)
    elif action_plus_random_sampling:
        if env_id is not None:
            import gym
            env = gym.make(env_id)
        else:
            env = env_class(**env_kwargs)
            env =  ImageEnv(
                env, imsize,
                transpose=True,
                init_camera=init_camera,
                normalize=True,
            )
        action_sampled_data = int(N*ratio_action_sample_to_random)
        dataset = np.zeros((N, imsize * imsize * 3), dtype=np.uint8)
        print('Action Space Sampling')
        for i in range(action_sampled_data):
            goal = env.sample_goal()
            env.set_to_goal(goal)
            img = env._get_flat_img()
            dataset[i, :] = unormalize_image(img)
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
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
            goal = env.sample_goal()
            env.set_to_goal_angle(goal['state_desired_goal'])
            img = env._get_flat_img()
            dataset[i, :] = unormalize_image(img)
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
        env._wrapped_env.min_y_pos = .5
        info['env'] = env
    else:
        raise NotImplementedError()
    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(
        500,
        use_cached=False,
        show=True,
        env_kwargs=dict(
            min_y_pos=.4,
        ),
        action_plus_random_sampling=True,
        ratio_action_sample_to_random=1/2,
    )
