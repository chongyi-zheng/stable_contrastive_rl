import numpy as np
from PIL import Image

from rlkit.envs.images import Renderer


class EnvRenderer(Renderer):
    # TODO: switch to env.render interface
    def __init__(
            self,
            init_camera=None,
            normalize_image=True,  # most gym envs output uint8
            create_image_format='HWC',
            **kwargs
    ):
        """Render an image."""
        super().__init__(
            normalize_image=normalize_image,
            create_image_format=create_image_format,
            **kwargs)
        self._init_camera = init_camera
        self._camera_is_initialized = False

    def _create_image(self, env):
        if not self._camera_is_initialized and self._init_camera is not None:
            env.initialize_camera(self._init_camera)
            self._camera_is_initialized = True

        return env.get_image(
            width=self.width,
            height=self.height,
        )


class GymEnvRenderer(EnvRenderer):
    def _create_image(self, env):
        if not self._camera_is_initialized and self._init_camera is not None:
            env.initialize_camera(self._init_camera)
            self._camera_is_initialized = True

        return env.render(
            mode='rgb_array', width=self.width, height=self.height
        )


class GymSimRenderer(EnvRenderer):
    def _create_image(self, env):
        if not self._camera_is_initialized and self._init_camera is not None:
            env.initialize_camera(self._init_camera)
            self._camera_is_initialized = True

        return self._get_image(env, self.width, self.height, camera_name=None)

    def _get_image(self, env, width=84, height=84, camera_name=None):
        return env.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )[::-1,:,:]
