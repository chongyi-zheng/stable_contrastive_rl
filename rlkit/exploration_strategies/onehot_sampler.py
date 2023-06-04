import numpy as np

from rlkit.exploration_strategies.base import RawExplorationStrategy
from rlkit.util.np_util import to_onehot


class OneHotSampler(RawExplorationStrategy):
    """
    Given a probability distribution over a set of discrete action, this ES
    samples one value and returns a one-hot vector.
    """
    def __init__(self, laplace_weight=0., softmax=False, **kwargs):
        self.laplace_weight = laplace_weight
        self.softmax = softmax or self.laplace_weight > 0.

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        return self.get_action_from_raw_action(action), agent_info

    def get_action_from_raw_action(self, action, **kwargs):
        num_values = len(action)
        elements = np.arange(num_values)
        action += self.laplace_weight
        if self.softmax:
            action = np.exp(action)
            action /= sum(action)
        number = np.random.choice(elements, p=action)
        return to_onehot(number, num_values)
