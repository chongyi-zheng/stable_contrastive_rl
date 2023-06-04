from collections import OrderedDict
from random import randint

import tensorflow as tf
import numpy as np
from gym.spaces import Box

from rlkit.envs.supervised_learning_env import RecurrentSupervisedLearningEnv
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.np_util import np_print_options
from rlkit.pythonplusplus import clip_magnitude
from rlkit.core import logger


def _generate_sign():
    return 2*randint(0, 1) - 1


class HighLow(RecurrentSupervisedLearningEnv):
    def __init__(self, horizon, give_time):
        assert horizon > 0
        self._horizon = horizon
        self.give_time = give_time
        self._t = 0
        self._sign = _generate_sign()
        self._action_space = Box(np.array([-1]), np.array([1]))
        if self.give_time:
            self._observation_space = Box(
                np.array([-1, 0]), np.array([1, self.horizon])
            )
        else:
            self._observation_space = Box(np.array([-1]), np.array([1]))

        # For rendering
        self._last_reward = None
        self._last_action = None
        self._last_t = None

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        self._t = 0
        self._sign = _generate_sign()
        if self.give_time:
            return np.array([self._sign, self._t])
        else:
            return np.array([self._sign])

    def step(self, action):
        self._last_t = self._t
        self._t += 1
        done = self._t == self.horizon
        action = max(-1, min(action, 1))
        if done:
            reward = float(action * self._sign)
        else:
            reward = 0
        if self.give_time:
            observation = np.array([0, self._t])
        else:
            observation = np.array([0])
        # To cheat:
        # observation = np.array([self._sign])
        info = self._get_info_dict()
        self._last_reward = reward
        self._last_action = action
        return observation, reward, done, info

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def _get_info_dict(self):
        return {
            'target_number': self._sign,
            'time': self._t,
        }

    def get_tf_loss(self, observations, actions, target_labels, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        target_labels_float = tf.cast(target_labels, tf.float32)
        assert target_labels_float.get_shape().is_compatible_with(
            actions.get_shape()
        )
        return actions * target_labels_float

    def log_diagnostics(self, paths):
        final_values = []
        final_unclipped_rewards = []
        final_rewards = []
        for path in paths:
            final_value = path["actions"][-1][0]
            final_values.append(final_value)
            score = path["observations"][0][0] * final_value
            final_unclipped_rewards.append(score)
            final_rewards.append(clip_magnitude(score, 1))

        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'Final Value',
            final_values,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Unclipped Final Rewards',
            final_unclipped_rewards,
        ))
        last_statistics.update(create_stats_ordered_dict(
            'Final Rewards',
            final_rewards,
        ))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return final_unclipped_rewards

    @staticmethod
    def get_extra_info_dict_from_batch(batch):
        return dict(
            target_numbers=batch['target_numbers'],
            times=batch['times'],
        )

    @staticmethod
    def get_flattened_extra_info_dict_from_subsequence_batch(batch):
        target_numbers = batch['target_numbers']
        times = batch['times']
        flat_target_numbers = target_numbers.flatten()
        flat_times = times.flatten()
        return dict(
            target_numbers=flat_target_numbers,
            times=flat_times,
        )

    @staticmethod
    def get_last_extra_info_dict_from_subsequence_batch(batch):
        target_numbers = batch['target_numbers']
        times = batch['times']
        last_target_numbers = target_numbers[:, -1]
        last_times = times[:, -1]
        return dict(
            target_numbers=last_target_numbers,
            times=last_times,
        )

    def render(self):
        logger.push_prefix("HighLow(sign={0})\t".format(self._sign))
        if self._last_action is None:
            logger.log("No action taken.")
        else:
            if self._last_t == 0:
                logger.log("--- New Episode ---")
            logger.push_prefix("t={0}\t".format(self._last_t))
            with np_print_options(precision=4, suppress=False):
                logger.log("Action: {0}".format(
                    self._last_action,
                ))
            logger.log("Reward: {0}".format(
                self._last_reward,
            ))
            logger.pop_prefix()
        logger.pop_prefix()

    """
    RecurrentSupervisedLearningEnv functions
    """
    @property
    def target_dim(self):
        return 1

    @property
    def feature_dim(self):
        return 1

    def get_batch(self, batch_size):
        targets = 2 * np.random.randint(
            low=0,
            high=2,
            size=batch_size,
        ) - 1
        targets = np.expand_dims(targets, 1)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, 0, :] = targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        # targets = np.expand_dims(targets, 2)
        Y[:, -1, :] = targets
        return X, Y

    @property
    def sequence_length(self):
        return self._horizon
