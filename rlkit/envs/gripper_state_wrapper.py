# import Dict

import numpy as np
from gym.spaces import Box, Dict

from rlkit.envs.wrappers import ProxyEnv
from roboverse.bullet.misc import quat_to_deg


def process_gripper_state(gripper_state):
    gripper_pos = gripper_state[:3]
    gripper_ori = quat_to_deg(gripper_state[3:7]) / 360.0
    gripper_tips_distance = gripper_state[7:8]
    return np.concatenate([gripper_pos, gripper_ori, gripper_tips_distance], axis=0)


class GripperStateWrappedEnv(ProxyEnv):
    def __init__(self,
                 wrapped_env,
                 state_key,
                 step_keys_map=None,
                 ):
        super().__init__(wrapped_env)
        self.state_key = state_key
        self.gripper_state_size = 7
        gripper_state_space = Box(
            -1 * np.ones(self.gripper_state_size),
            1 * np.ones(self.gripper_state_size),
            dtype=np.float32,
        )

        if step_keys_map is None:
            step_keys_map = {}
        self.step_keys_map = step_keys_map
        spaces = self.wrapped_env.observation_space.spaces
        for value in self.step_keys_map.values():
            spaces[value] = gripper_state_space
        self.observation_space = Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        for key in self.step_keys_map:
            value = self.step_keys_map[key]
            gripper_state = obs[self.state_key]
            process_gripper_state(gripper_state)
            obs[value] = process_gripper_state(gripper_state)
        obs = {**obs, **self.reset_obs}
        return obs
