import numpy as np

from rlkit.policies.base import Policy


class ActionRepeatPolicy(Policy):
    """
    General policy interface.
    """
    def __init__(self, policy, repeat_prob=.5):
        self._policy = policy
        self._repeat_prob = repeat_prob
        self._last_action = None

    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        action = self._policy.get_action(observation)
        if (
                self._last_action is not None
                and np.random.uniform() <= self._repeat_prob
        ):
            action = self._last_action
        self._last_action = action
        return self._last_action

    def reset(self):
        self._last_action = None