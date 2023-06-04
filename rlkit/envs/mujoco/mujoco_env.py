import os
from os import path

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env

from rlkit.envs.env_utils import get_asset_full_path


class MujocoEnv(mujoco_env.MujocoEnv):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """
    def __init__(
            self,
            model_path,
            frame_skip=1,
            model_path_is_local=True,
            automatically_set_obs_and_action_space=False,
    ):
        if model_path_is_local:
            model_path = get_asset_full_path(model_path)
        if automatically_set_obs_and_action_space:
            mujoco_env.MujocoEnv.__init__(self, model_path, frame_skip)
        else:
            """
            Code below is copy/pasted from MujocoEnv's __init__ function.
            """
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            self.frame_skip = frame_skip
            self.model = mujoco_py.load_model_from_path(fullpath)
            self.sim = mujoco_py.MjSim(self.model)
            self.data = self.sim.data
            self.viewer = None

            self.metadata = {
                'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': int(np.round(1.0 / self.dt))
            }

            self.init_qpos = self.sim.data.qpos.ravel().copy()
            self.init_qvel = self.sim.data.qvel.ravel().copy()
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high)

            high = np.inf*np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)

            self.seed()

    def log_diagnostics(self, paths):
        pass
