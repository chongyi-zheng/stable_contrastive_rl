import numpy as np
import torch
# from scipy import spatial

import rlkit.torch.pytorch_util as ptu
from rlkit.util.io import load_local_or_remote_file


class GoalReachingRewardFn:
    def __init__(self,
                 env,
                 obs_type='latent',
                 reward_type='dense',
                 epsilon=1.0,
                 cosine_reward_thresh=0.95,
                 terminate_episode=False,  # TODO
                 use_pretrained_reward_classifier_path=False,
                 pretrained_reward_classifier_path='',
                 ):
        self.obs_type = obs_type

        if self.obs_type == 'image':
            self.obs_key = 'image_observation'
            self.goal_key = 'image_desired_goal'
        elif self.obs_type == 'latent':
            self.obs_key = 'latent_observation'
            self.goal_key = 'latent_desired_goal'
        elif self.obs_type == 'vib':
            self.obs_key = 'vib_observation'
            self.goal_key = 'vib_desired_goal'
        elif self.obs_type == 'state':
            self.obs_key = 'state_observation'
            self.goal_key = 'state_desired_goal'
        else:
            raise ValueError('Unrecognized obs_type: %r' % (self.obs_type))

        self.env = env
        self.reward_type = reward_type
        self.epsilon = epsilon
        self.cosine_reward_thresh = cosine_reward_thresh
        self.terminate_episode = terminate_episode

        if reward_type == 'classifier':
            self.reward_classifier = load_local_or_remote_file(
                pretrained_reward_classifier_path)
            self.sigmoid = torch.nn.Sigmoid()

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = self.process(next_states[self.obs_key])
        c = self.process(contexts[self.goal_key])

        terminal = np.zeros((s.shape[0], ), dtype=np.uint8)

        if self.reward_type is None or self.reward_type == 'none':
            reward = np.zeros_like(actions[..., 0:1], dtype=np.float32)

        elif self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)

        elif self.reward_type == 'sparse':
            dist = np.linalg.norm(s - c, axis=1)
            success = dist < self.epsilon
            reward = success - 1
            if self.terminate_episode:
                terminal = np.array(success, dtype=np.uint8)

        elif self.reward_type == 'cosine':
            assert self.obs_type == 'vib'
            similarity = (
                np.sum(s * c, -1) /
                (np.linalg.norm(s, axis=-1) * np.linalg.norm(c, axis=-1)
                 + 1e-12))
            success = similarity >= self.cosine_reward_thresh
            reward = success - 1
            if self.terminate_episode:
                terminal = np.array(success, dtype=np.uint8)

        elif self.reward_type == 'progress':
            s_tm1 = self.process(states[self.obs_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            reward = sd_tm1 - sd_t

        elif self.reward_type == 'highlevel':
            assert self.obs_type == 'state'
            reward = self.env.compute_reward(
                states, actions, next_states, contexts)

        elif self.reward_type == 'classifier':
            s = ptu.from_numpy(s)
            s = s.view(s.shape[0], 5, 12, 12)
            c = ptu.from_numpy(c)
            c = c.view(c.shape[0], 5, 12, 12)
            pred = self.sigmoid(self.reward_classifier(s, c))
            pred = ptu.get_numpy(pred)[..., 0]
            reward = pred - 1.0

        elif self.reward_type in ['sp', 'sparse_progress']:
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            sparse_reward = success - 1

            s_tm1 = self.process(states[self.obs_key])
            sd_tm1 = np.square(np.linalg.norm(s_tm1 - c, axis=1))
            sd_t = np.square(np.linalg.norm(s - c, axis=1))
            progress_reward = sd_tm1 - sd_t

            reward = sparse_reward + 0.1 * progress_reward

        else:
            raise ValueError(self.reward_type)

        # print(reward.shape, terminal.shape)
        return reward, terminal
