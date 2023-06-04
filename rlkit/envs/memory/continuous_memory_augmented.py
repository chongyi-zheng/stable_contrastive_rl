import numpy as np

from rllab.envs.base import Env
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from cached_property import cached_property


def split_flat_product_space_into_components_n(product_space, xs):
    """
    Split up a flattened block into its components

    :param product_space: ProductSpace instance
    :param xs: N x flat_dim
    :return: list of (N x component_dim)
    """
    dims = [c.flat_dim for c in product_space.components]
    return np.split(xs, np.cumsum(dims)[:-1], axis=-1)


class ContinuousMemoryAugmented(ProxyEnv):
    """
    An environment that wraps another environments and adds continuous memory
    states/actions.
    """

    def __init__(
            self,
            env: Env,
            num_memory_states=10,
            max_magnitude=1e6,
    ):
        super().__init__(env)
        self._num_memory_states = num_memory_states
        assert max_magnitude > 0
        self._max_magnitude = max_magnitude
        self._memory_state = np.zeros(self._num_memory_states)
        self._action_space = Product(
            env.action_space,
            self.memory_state_space,
        )
        self._observation_space = Product(
            env.observation_space,
            self.memory_state_space,
        )

    @cached_property
    def memory_state_space(self):
        return Box(-self._max_magnitude * np.ones(self._num_memory_states),
                   self._max_magnitude * np.ones(self._num_memory_states))

    def reset(self):
        self._memory_state = np.zeros(self._num_memory_states)
        env_obs = self._wrapped_env.reset()
        return env_obs, self._memory_state

    def step(self, action):
        """
        :param action: An unflattened action, i.e. action = (environment
        action,
        memory write) tuple.
        :return: An unflattened observation.
        """
        env_action, memory_state = action
        observation, reward, done, info = self._wrapped_env.step(env_action)
        return (
            # Squeeze the memory state since the returned next_observation
            # should be flat.
            (observation, memory_state.squeeze()),
            reward,
            done,
            info
        )

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def memory_dim(self):
        return self._num_memory_states

    def _strip_path(self, path):
        path = path.copy()
        actions = path['actions']
        env_actions = split_flat_product_space_into_components_n(
            self.action_space,
            actions
        )[0]
        path['actions'] = env_actions

        observations = path['observations']
        env_obs = split_flat_product_space_into_components_n(
            self.observation_space,
            observations
        )[0]
        path['observations'] = env_obs

        return path

    def log_diagnostics(self, paths):
        non_memory_paths = [self._strip_path(path) for path in paths]
        return self._wrapped_env.log_diagnostics(non_memory_paths)

    def get_tf_loss(self, observations, actions, **kwargs):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        return self._wrapped_env.get_tf_loss(observations[0], actions[0],
                                             **kwargs)

    def get_extra_info_dict_from_batch(self, batch):
        return self._wrapped_env.get_extra_info_dict_from_batch(batch)

    def get_flattened_extra_info_dict_from_subsequence_batch(self, batch):
        return self._wrapped_env.get_flattened_extra_info_dict_from_subsequence_batch(batch)

    def get_last_extra_info_dict_from_subsequence_batch(self, batch):
        return self._wrapped_env.get_last_extra_info_dict_from_subsequence_batch(batch)
