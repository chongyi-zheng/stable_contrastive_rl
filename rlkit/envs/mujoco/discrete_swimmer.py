import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces

from rlkit.core.eval_util import get_stat_in_paths
from rlkit.core import logger
import itertools


class DiscreteSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, num_bins=5, ctrl_cost_coeff=0.0001,
                 reward_position=False):
        self.num_bins = num_bins
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.reward_position = reward_position
        utils.EzPickle.__init__(self, num_bins=num_bins)
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        joint0_range = np.linspace(low[0], high[0], num_bins)
        joint1_range = np.linspace(low[1], high[1], num_bins)
        self.idx_to_continuous_action = list(itertools.product(joint0_range, joint1_range))
        self.action_space = spaces.Discrete(len(self.idx_to_continuous_action))

    def _step(self, a):
        if not self.action_space or not self.action_space.contains(a):
            continuous_action = a
        else:
            continuous_action = self.idx_to_continuous_action[a]

        self.do_simulation(continuous_action, self.frame_skip)
        if self.reward_position:
            reward_fwd = self.get_body_com("torso")[0]
        else:
            reward_fwd = self.get_body_comvel("torso")[0]
        reward_ctrl = - self.ctrl_cost_coeff * np.square(
            continuous_action
        ).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def log_diagnostics(self, paths):
        reward_fwd = get_stat_in_paths(paths, 'env_infos', 'reward_fwd')
        reward_ctrl = get_stat_in_paths(paths, 'env_infos', 'reward_ctrl')

        logger.record_tabular('AvgRewardDist', np.mean(reward_fwd))
        logger.record_tabular('AvgRewardCtrl', np.mean(reward_ctrl))
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)


def main():
    env = DiscreteSwimmerEnv()
    for i in range(10000):
        env.step(env.action_space.sample())
        env.render()


if __name__ == '__main__':
    main()