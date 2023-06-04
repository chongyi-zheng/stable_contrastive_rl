import numpy as np

from rlkit.envs.mujoco.mujoco_env import MujocoEnv

TARGET = np.array([0.2, 0])


class TwoDPointRandomInit(MujocoEnv):
    def __init__(self):
        self.init_serialization(locals())
        super().__init__('twod_point.xml')

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        dist = np.linalg.norm(pos - TARGET)
        reward = - (dist + 1e-2*np.linalg.norm(a))
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.np_random.uniform(size=self.model.nq, low=-0.25, high=0.25)
        qvel = self.np_random.uniform(size=self.model.nv, low=-0.25, high=0.25)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos]).ravel()

    def viewer_setup(self):
        pass
