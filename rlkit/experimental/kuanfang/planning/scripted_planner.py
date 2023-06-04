import numpy as np
import torch

from rlkit.torch import pytorch_util as ptu
from rlkit.util.io import load_local_or_remote_file

from rlkit.experimental.kuanfang.planning.planner import Planner
from rlkit.experimental.kuanfang.utils.logging import logger as logging


class ScriptedPlanner(Planner):

    def __init__(
        self,
        model,
        path,
        **kwargs
    ):

        kwargs['buffer_size'] = 0
        super().__init__(
            model=model,
            **kwargs)

        self._data = load_local_or_remote_file(path)
        self._num_samples = self._data['image_plan'].shape[0]
        self._max_steps = self._data['image_plan'].shape[1]

        self._init_images = self._data['initial_image_observation']
        self._goal_images = self._data['image_desired_goal']
        self._plan_images = self._data['image_plan']

        if self.encoding_type in ['vqvae', 'vib']:
            self._init_latents = self.vqvae.encode_np(
                self._init_images)
            self._goal_latents = self.vqvae.encode_np(
                self._goal_images)

            if self.encoding_type == 'vib':
                self._init_latents = self.obs_encoder.encode_np(
                    self._init_latents)
                self._goal_latents = self.obs_encoder.encode_np(
                    self._goal_latents)

            self._plan_latents = []
            for i in range(self._num_samples):
                plan_latents_i = self.vqvae.encode_np(
                    self._plan_images[i])
                if self.encoding_type == 'vib':
                    plan_latents_i = self.obs_encoder.encode_np(
                        plan_latents_i)

                self._plan_latents.append(plan_latents_i)
            self._plan_latents = np.stack(self._plan_latents, 0)

        else:
            self._init_latents = None
            self._goal_latents = None
            self._plan_latents = None

    @torch.no_grad()
    def __call__(self, init_state, goal_state, num_steps):
        assert num_steps <= self._max_steps

        goal_state_np = ptu.get_numpy(goal_state.flatten())

        if self.encoding_type in ['vqvae', 'vib']:
            scripted_goals = self._goal_latents
            scripted_plans = self._plan_latents
        else:
            scripted_goals = self._goal_images
            scripted_plans = self._plan_images

        dists = np.linalg.norm(
            goal_state_np[None, ...] - scripted_goals, axis=-1)
        top_ind = np.argmin(dists)
        top_dist = dists[top_ind]

        plan = np.concatenate(
            [scripted_plans[top_ind, :num_steps - 1],
             scripted_plans[top_ind, -1][None, ...]],
            axis=0)

        if self.encoding_type == 'vqvae':
            plan = np.reshape(
                plan,
                [num_steps,
                 self.vqvae.embedding_dim,
                 self.vqvae.root_len,
                 self.vqvae.root_len])
        elif self.encoding_type == 'vib':
            plan = np.reshape(plan, [num_steps, -1])

        plan = ptu.from_numpy(plan)

        info = {
            'top_step': -1,
            'top_cost': top_dist,
        }

        return plan, info


class RandomChoicePlanner(Planner):

    def __init__(
        self,
        model,
        path,
        debug=False,
        batch_size=128,
        uniform=False,
        **kwargs,
    ):

        super().__init__(
            model=model,
            debug=debug,
            **kwargs)

        self._uniform = uniform

        self._data = load_local_or_remote_file(path)

        self._images = self._data['image_observation']
        self._num_samples = int(self._images.shape[0])

        self._init_images = self._images[:, :-1].reshape(
            (self._images.shape[0] * (self._images.shape[1] - 1),
             self._images.shape[2],)
        )

        self._goal_images = self._images[:, 1:].reshape(
            (self._images.shape[0] * (self._images.shape[1] - 1),
             self._images.shape[2],)
        )

        # Encode.
        logging.info('Encoding %d init images...', self._num_samples)
        i_start = 0
        self._init_latents = []
        while i_start < self._num_samples:
            i_end = min(self._num_samples, i_start + batch_size)
            latents = self.vqvae.encode_np(
                self._init_images[i_start:i_end])
            i_start = i_end
            self._init_latents.append(latents)
        self._init_latents = np.concatenate(self._init_latents, axis=0)
        logging.info('Done.')

        logging.info('Encoding %d goal images...', self._num_samples)
        i_start = 0
        self._goal_latents = []
        while i_start < self._num_samples:
            i_end = min(self._num_samples, i_start + batch_size)
            latents = self.vqvae.encode_np(
                self._goal_images[i_start:i_end])
            i_start = i_end
            self._goal_latents.append(latents)
        self._goal_latents = np.concatenate(self._goal_latents, axis=0)
        logging.info('Done.')

    @torch.no_grad()
    def __call__(self, init_state, goal_state, num_steps):
        assert num_steps == 1

        init_state = self._process_state(init_state)
        goal_state = self._process_state(goal_state)

        if self._uniform:
            inds = np.random.choice(self._num_samples, num_steps)

        else:
            init_state_np = ptu.get_numpy(init_state.flatten())
            dists = np.linalg.norm(
                init_state_np[None, ...] - self._init_latents, axis=-1)
            top_ind = np.argmin(dists)
            inds = np.array([top_ind])

            top_dist = dists[top_ind]
            print('top_dist: ', top_dist, 'mean_dist: ', np.mean(dists))

        plan = self._goal_latents[inds]
        plan = ptu.from_numpy(plan)

        info = {}

        return plan, info
