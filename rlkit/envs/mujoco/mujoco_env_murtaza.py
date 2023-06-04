import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip, init_viewer=False):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.viewer = None

        if init_viewer:
            self._viewer_bot.set_model(self.model)
            self._set_cam_position(self._viewer_bot, self.cam_pos)
            if init_viewer > 1:
                print("viewer 2 bring set up")
                self._viewer_bot2.set_model(self.model)
                self._set_cam_position(self._viewer_bot2, self.cam_pos2)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self._seed()