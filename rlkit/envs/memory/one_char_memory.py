import numpy as np
from sklearn.metrics import log_loss
from random import randint, choice

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.np_util import np_print_options, softmax
from collections import OrderedDict
from rlkit.pythonplusplus import clip_magnitude
from rllab.envs.base import Env
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from rlkit.core import logger
from rlkit.envs.supervised_learning_env import RecurrentSupervisedLearningEnv
from cached_property import cached_property
import tensorflow as tf


class OneCharMemory(Env, RecurrentSupervisedLearningEnv):
    """
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    The goal of the agent is to output zero for all time steps, and then
    output `X` in the last time step.

    Both the actions and observations are represented as probability vectors.
    There are `n` different values that `X` can take on (excluding 0),
    so the probability vector's dimension is n+1.

    The reward is the negative cross-entropy loss between the target one-hot
    vector and the probability vector outputted by the agent. Furthermore, the
    reward for the last time step is multiplied by `reward_for_remember`.
    """

    def __init__(
            self,
            n=2,
            num_steps=10,
            reward_for_remembering=1,
            max_reward_magnitude=1,
            softmax_action=False,
            zero_observation=False,
            output_target_number=False,
            output_time=False,
            episode_boundary_flags=False,
    ):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the policy needs to output before the
        episode ends. AKA the horizon.

        So, the policy will see num_steps+1 total observations
        if you include the observation returned when reset() is called. The
        last observation will be the "terminal state" observation.
        :param reward_for_remembering: The reward bonus for remembering the
        number. This number is added to the usual reward if the correct
        number has the maximum probability.
        :param max_reward_magnitude: Clip the reward magnitude to this value.
        :param softmax_action: If true, put the action through a softmax.
        :param zero_observation: If true, all observations after the first will
        be just zeros (NOT the zero one-hot).
        :param episode_boundary_flags: If true, add a boolean flag to the
        observation for whether or not the episode is starting or terminating.
        """
        assert max_reward_magnitude >= reward_for_remembering
        self.num_steps = num_steps
        self.n = n
        self._onehot_size = n + 1
        """
        Time step for the NEXT observation to be returned.

        env = OneCharMemory()  # t == 0 after this line
        obs = env.reset()      # t == 1
        _ = env.step()         # t == 2
        _ = env.step()         # t == 3
        ...
        done = env.step()[2]   # t == num_steps and done == False
        done = env.step()[2]   # t == num_steps+1 and done == True
        """
        self._t = 0
        self._reward_for_remembering = reward_for_remembering
        self._max_reward_magnitude = max_reward_magnitude
        self._softmax_action = softmax_action
        self._zero_observation = zero_observation
        self._output_target_number = output_target_number
        self._output_time = output_time
        self._episode_boundary_flags = episode_boundary_flags

        self._action_space = Box(
            np.zeros(self._onehot_size),
            np.ones(self._onehot_size)
        )
        obs_low = np.zeros(self._onehot_size)
        obs_high = np.zeros(self._onehot_size)
        if self._output_target_number:
            obs_low = np.hstack((obs_low, [0]))
            obs_high = np.hstack((obs_high, [self.n]))
        if self._output_time:
            obs_low = np.hstack((obs_low, [0] * (self.num_steps + 1)))
            obs_high = np.hstack((obs_high, [1] * (self.num_steps + 1)))
        if self._episode_boundary_flags:
            obs_low = np.hstack((obs_low, [0] * 2))
            obs_high = np.hstack((obs_high, [1] * 2))
        self._observation_space = Box(obs_low, obs_high)

        self._target_number = None

        # For rendering
        self._last_reward = None
        self._last_action = None
        self._last_t = None

    def step(self, action):
        if self._softmax_action:
            action = softmax(action)
        # Reset gives the first observation, so only return 0 in step.
        observation = self._get_next_observation()
        done = self.done
        info = self._get_info_dict()
        reward = self._compute_reward(done, action)

        self._last_t = self._t
        self._t += 1

        self._last_reward = reward
        self._last_action = action
        return observation, reward, done, info

    @property
    def done(self) -> bool:
        return self._t == self.num_steps

    @property
    def will_take_last_action(self) -> bool:
        return self._t + 1 == self.num_steps

    def _get_info_dict(self):
        return {
            'target_number': self._target_number,
            'time': self._t,
            'reward_for_remembering': self._reward_for_remembering,
        }

    def _compute_reward(self, done, action):
        try:
            if done:
                reward = -log_loss(self._get_target_onehot(), action)
                if np.argmax(action) == self._target_number:
                    reward += self._reward_for_remembering
            else:
                reward = -log_loss(self.zero, action)
            # if reward == -np.inf:
            #     reward = -self._max_reward_magnitude
            # if reward == np.inf or np.isnan(reward):
            #     reward = self._max_reward_magnitude
        except ValueError as e:
            raise e
            # reward = -self._max_reward_magnitude
        reward = clip_magnitude(reward, self._max_reward_magnitude)
        return reward

    @cached_property
    def zero(self):
        z = np.zeros(self._onehot_size)
        z[0] = 1
        return z

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target_number = randint(1, self.n)
        self._t = 0
        observation = self._get_first_observation()
        self._t += 1
        return observation

    def _get_first_observation(self):
        return self._get_observation(self._target_number, first=True)

    def _get_next_observation(self):
        return self._get_observation(0)

    def _get_observation(self, number, first=False):
        observation = special.to_onehot(number, self._onehot_size)
        if not first and self._zero_observation:
            observation = np.zeros(self._onehot_size)
        if self._output_target_number:
            observation = np.hstack((observation, [self._target_number]))
        if self._output_time:
            time = special.to_onehot(self._t, self.num_steps + 1)
            observation = np.hstack((observation, time))
        if self._episode_boundary_flags:
            observation = np.hstack((
                observation,
                [int(first), int(self.will_take_last_action)],
            ))
        return observation

    def _get_target_onehot(self):
        return special.to_onehot(self._target_number, self._onehot_size)

    @property
    def observation_space(self):
        return self._observation_space

    def get_batch(self, batch_size):
        targets = np.random.randint(
            low=1,
            high=self.n+1,
            size=batch_size,
        )
        onehot_targets = special.to_onehot_n(targets, self.feature_dim)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, :, 0] = 1  # make the target 0
        X[:, 0, :] = onehot_targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        Y[:, :, 0] = 1  # make the target 0
        Y[:, -1, :] = onehot_targets
        return X, Y

    @property
    def feature_dim(self):
        return self.n + 1

    @property
    def target_dim(self):
        return self.n + 1

    @property
    def sequence_length(self):
        return self.horizon

    @overrides
    def render(self):
        logger.push_prefix("OneCharMemory(n={0})\t".format(self._target_number))
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

    def log_diagnostics(self, paths):
        target_onehots = []
        for path in paths:
            first_observation = path["observations"][0][:self.n+1]
            target_onehots.append(first_observation)

        final_predictions = []  # each element has shape (dim)
        nonfinal_predictions = []  # each element has shape (seq_length-1, dim)
        for path in paths:
            actions = path["actions"]
            if self._softmax_action:
                actions = softmax(actions, axis=-1)
            final_predictions.append(actions[-1])
            nonfinal_predictions.append(actions[:-1])
        nonfinal_predictions_sequence_dimension_flattened = np.vstack(
            nonfinal_predictions
        )  # shape = N X dim
        nonfinal_prob_zero = [softmax[0] for softmax in
                              nonfinal_predictions_sequence_dimension_flattened]
        final_probs_correct = []
        for final_prediction, target_onehot in zip(final_predictions,
                                                   target_onehots):
            correct_pred_idx = np.argmax(target_onehot)
            final_probs_correct.append(final_prediction[correct_pred_idx])
        final_prob_zero = [softmax[0] for softmax in final_predictions]

        last_statistics = OrderedDict()
        last_statistics.update(create_stats_ordered_dict(
            'Final P(correct)',
            final_probs_correct))
        last_statistics.update(create_stats_ordered_dict(
            'Non-final P(zero)',
            nonfinal_prob_zero))
        last_statistics.update(create_stats_ordered_dict(
            'Final P(zero)',
            final_prob_zero))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return final_probs_correct

    def get_tf_loss(self, observations, actions, target_labels,
                    return_expected_reward=False):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        target_labels_float = tf.cast(target_labels, tf.float32)
        if return_expected_reward:
            assert target_labels_float.get_shape().is_compatible_with(
                actions.get_shape()
            )
            prob_correct = target_labels_float * actions
            return 2 * tf.reduce_sum(prob_correct, axis=1, keep_dims=True) - 1
        cross_entropy = target_labels_float * tf.log(actions)
        return tf.reduce_sum(cross_entropy, axis=1, keep_dims=True)

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


class OneCharMemoryEndOnly(OneCharMemory):
    """
    Don't reward or penalize outputs other than the last output. Then,
    only give a 1 or 0.
    """
    def _compute_reward(self, done, action):
        if done:
            if np.argmax(action) == self._target_number:
                return self._reward_for_remembering
            else:
                return - self._reward_for_remembering
        return 0


class OneCharMemoryEndOnlyLogLoss(OneCharMemory):
    """
    Don't reward or penalize outputs other than the last output. Then,
    give the usual reward.
    """
    def _compute_reward(self, done, action):
        if done:
            return super()._compute_reward(done, action)
        return 0


class OneCharMemoryOutputRewardMag(OneCharMemoryEndOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_values = list(
            range(1, int(self._max_reward_magnitude) + 1)
        )

        low = np.zeros(self._onehot_size + 1)
        low[-1] = - self._max_reward_magnitude
        high = np.ones(self._onehot_size + 1)
        high[-1] = self._max_reward_magnitude

        self._observation_space = Box(low, high)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = np.hstack((observation, 0))
        return observation, reward, done, info

    def reset(self):
        self._reward_for_remembering = choice(self._reward_values)
        self._target_number = randint(1, self.n)
        self._t = 0
        first_observation = self._get_first_observation()
        return np.hstack((first_observation, self._reward_for_remembering))

    def log_diagnostics(self, paths):
        # Take out the extra reward observation
        for path in paths:
            path["observations"] = path["observations"][:, :-1]
        return super().log_diagnostics(paths)
