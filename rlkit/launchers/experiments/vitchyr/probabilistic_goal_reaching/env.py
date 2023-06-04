import gym
import numpy as np

from multiworld.envs.mujoco.classic_mujoco.ant import AntFullPositionGoalEnv


class NormalizeAntFullPositionGoalEnv(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, AntFullPositionGoalEnv)
        super().__init__(env)
        self.qpos_weights = 1. / env.presampled_qpos.std(axis=0)
        self.qvel_weights = 1. / env.presampled_qvel.std(axis=0)
        self._ob_weights = np.concatenate((
            self.qpos_weights,
            self.qvel_weights,
        ))

    def reset(self):
        ob = super().reset()
        new_ob = self._create_new_ob(ob)
        return new_ob

    def step(self, action):
        ob, *other = self.env.step(action)
        new_ob = self._create_new_ob(ob)
        output = (new_ob, *other)
        return output

    def _create_new_ob(self, ob):
        new_ob = {
            'observation': ob['observation'] * self._ob_weights,
            'achieved_goal': ob['achieved_goal'] * self.qpos_weights,
            'desired_goal': ob['desired_goal'] * self.qpos_weights,
        }
        return new_ob
