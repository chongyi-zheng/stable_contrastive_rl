
import os.path as osp
import time
import cv2
import numpy as np
from multiworld.core.image_env import ImageEnv, unormalize_image
from multiworld.envs.mujoco.cameras import sawyer_torque_reacher_camera
from multiworld.envs.mujoco.sawyer_reach_torque.sawyer_reach_torque_env import SawyerReachTorqueEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.util.io import local_path_from_s3_or_local_path
from rlkit.policies.simple import RandomPolicy

def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=False, imsize=84, show=False,
        dataset_path=None, env_class = SawyerReachTorqueEnv, env_kwargs=None, init_camera=sawyer_torque_reacher_camera,
):

    filename = "/tmp/sawyer_torque_data" + str(N) + ".npy"
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
        info['env'] = env
        policy = RandomPolicy(env.action_space)
        es = OUStrategy(action_space=env.action_space, theta=0)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        dataset = np.zeros((N, imsize * imsize * 3), dtype=np.uint8)
        for i in range(N):
            if i %50==0:
                print('Reset')
                env.reset_model()
                exploration_policy.reset()
            for _ in range(1):
                action = exploration_policy.get_action()[0]*1/10
                env.wrapped_env.step(
                    action
                )
            img = env._get_flat_img()
            dataset[i, :] = unormalize_image(img)
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
    generate_vae_dataset(1000, use_cached=False, show=True,
                         env_kwargs=dict(
                             keep_vel_in_obs=False,
                             use_safety_box=True,
                            ),
                         init_camera=sawyer_torque_reacher_camera,
                         )
