import numpy as np
from gym.spaces import Box, Dict
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import ProxyEnv
from rlkit.torch.core import PyTorchModule


class Encoder(object):
    def encode(self, input):
        """
        :param input:
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @property
    def space(self):
        raise NotImplementedError()


class EncoderFromNetwork(Encoder, PyTorchModule):
    def __init__(self, network):
        super(PyTorchModule, self).__init__()
        self._network = network
        self._output_size = network.output_size
        self._space = Box(-9999, 9999, (self._output_size,), dtype=np.float32)

    def encode(self, x):
        x_torch = ptu.from_numpy(x)
        embedding_torch = self._network(x_torch)
        return ptu.get_numpy(embedding_torch)

    @property
    def space(self):
        return self._space


class EncoderWrappedEnv(ProxyEnv, MultitaskEnv):
    """This class wraps an environment with an encoder.

    Reward is defined as distance in this latent space.
    """
    ENCODER_DISTANCE_REWARD = 'encoder_distance'
    VECTORIZED_ENCODER_DISTANCE_REWARD = 'vectorized_encoder_distance'
    ENV_REWARD = 'env'

    def __init__(
            self,
            wrapped_env,
            encoder: Encoder,
            encoder_input_prefix,
            key_prefix='encoder',
            reward_mode='encoder_distance',
    ):
        """

        :param wrapped_env:
        :param encoder:
        :param encoder_input_prefix:
        :param key_prefix:
        :param reward_mode:
         - 'encoder_distance': l1 distance in encoder distance
         - 'vectorized_encoder_distance': vectorized l1 distance in encoder
             distance, i.e. negative absolute value
         - 'env': use the wrapped env's reward
        """
        super().__init__(wrapped_env)
        if reward_mode not in {
            self.ENCODER_DISTANCE_REWARD,
            self.VECTORIZED_ENCODER_DISTANCE_REWARD,
            self.ENV_REWARD,
        }:
            raise ValueError(reward_mode)
        self._encoder = encoder
        self._encoder_input_obs_key = '{}_observation'.format(
            encoder_input_prefix)
        self._encoder_input_desired_goal_key = '{}_desired_goal'.format(
            encoder_input_prefix
        )
        self._encoder_input_achieved_goal_key = '{}_achieved_goal'.format(
            encoder_input_prefix
        )
        self._reward_mode = reward_mode
        spaces = self.wrapped_env.observation_space.spaces
        latent_space = self._encoder.space
        self._embedding_size = latent_space.low.size
        self._obs_key = '{}_observation'.format(key_prefix)
        self._desired_goal_key = '{}_desired_goal'.format(key_prefix)
        self._achieved_goal_key = '{}_achieved_goal'.format(key_prefix)
        self._distance_name = '{}_distance'.format(key_prefix)

        self._key_prefix = key_prefix
        self._desired_goal = {
            self._desired_goal_key: np.zeros_like(latent_space.sample())
        }
        spaces[self._obs_key] = latent_space
        spaces[self._desired_goal_key] = latent_space
        spaces[self._achieved_goal_key] = latent_space
        self.observation_space = Dict(spaces)
        self._goal_sampling_mode = 'env'

    def reset(self):
        obs = self.wrapped_env.reset()
        self._update_obs(obs)
        self._desired_goal = {
            self._desired_goal_key:
                self._encode_one(obs[self._encoder_input_desired_goal_key]),
            **self.wrapped_env.get_goal()
        }
        return obs

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        self._update_obs(obs)
        new_reward = self.compute_reward(
            action,
            obs,
        )
        self._update_info(info, obs, new_reward)
        return obs, new_reward, done, info

    def _update_obs(self, obs):
        encoded_obs = self._encode_one(obs[self._encoder_input_obs_key])
        obs[self._obs_key] = encoded_obs
        obs[self._achieved_goal_key] = encoded_obs
        obs[self._desired_goal_key] = self._desired_goal[self._desired_goal_key]
        obs['observation'] = encoded_obs
        obs['achieved_goal'] = encoded_obs
        obs['desired_goal'] = self._desired_goal[self._desired_goal_key]
        return obs

    def _update_info(self, info, obs, new_reward):
        achieved_goals = obs[self._achieved_goal_key]
        desired_goals = obs[self._desired_goal_key]
        dist = np.linalg.norm(desired_goals - achieved_goals, ord=1)
        info[self._distance_name] = dist

    """
    Multitask functions
    """

    def sample_goals(self, batch_size):
        if self._goal_sampling_mode == 'env':
            goals = self.wrapped_env.sample_goals(batch_size)
            latent_goals = self._encode(
                goals[self._encoder_input_desired_goal_key])
        else:
            raise RuntimeError("Invalid: {}".format(self._goal_sampling_mode))

        goals['desired_goal'] = latent_goals
        goals[self._desired_goal_key] = latent_goals
        return goals

    @property
    def goal_sampling_mode(self):
        return self._goal_sampling_mode

    @goal_sampling_mode.setter
    def goal_sampling_mode(self, mode):
        assert mode in [
            'env',
        ], "Invalid env mode: {}".format(mode)
        self._goal_sampling_mode = mode

    @property
    def goal_dim(self):
        return self._embedding_size

    def get_goal(self):
        return self._desired_goal

    def set_goal(self, goal):
        self._desired_goal = goal
        self.wrapped_env.set_goal(goal)

    def compute_reward(self, action, obs):
        actions = action[None]
        next_obs = {
            k: v[None] for k, v in obs.items()
        }
        reward = self.compute_rewards(actions, next_obs)
        return reward[0]

    def compute_rewards(self, actions, obs):
        achieved_goals = obs[self._achieved_goal_key]
        desired_goals = obs[self._desired_goal_key]
        if self._reward_mode == self.VECTORIZED_ENCODER_DISTANCE_REWARD:
            dist = np.abs(desired_goals - achieved_goals)
            rewards = - dist
        elif self._reward_mode == self.ENCODER_DISTANCE_REWARD:
            dist = np.linalg.norm(desired_goals - achieved_goals, ord=1, axis=1)
            rewards = - dist
        elif self._reward_mode == self.ENV_REWARD:
            rewards = self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise ValueError('iNvalid reward mode: {}'.format(
                self._reward_mode
            ))
        return rewards

    """
    Other functions
    """

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in [self._distance_name]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

    def _encode_one(self, ob):
        return self._encode(ob[None])[0]

    def _encode(self, obs):
        return self._encoder.encode(obs)
