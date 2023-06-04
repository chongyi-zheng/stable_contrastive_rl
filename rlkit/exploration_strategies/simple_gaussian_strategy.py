from rlkit.exploration_strategies.base import RawExplorationStrategy
import numpy as np


class SimpleGaussianStrategy(RawExplorationStrategy):
    """
    This strategy adds a constant Gaussian noise to the action taken by the
    deterministic policy.

    This is different from GaussianStrategy in that the sigma does not decay
    over time.
    """

    def __init__(self, action_space, sigma=1.0):
        assert len(action_space.shape) == 1
        super().__init__()
        self._sigma = sigma
        self._action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        return np.clip(
            action + np.random.normal(size=len(action))*self._sigma,
            self._action_space.low,
            self._action_space.high,
        )