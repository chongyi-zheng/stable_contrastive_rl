from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.networks import ConcatMlp


class DynamicsModel(ConcatMlp):
    def __init__(
            self,
            observation_dim,
            action_dim,
            obs_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            delta_normalizer: TorchFixedNormalizer=None,
            **kwargs
    ):
        super().__init__(
            input_size=observation_dim + action_dim,
            output_size=observation_dim,
            **kwargs
        )
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer
        self.delta_normalizer = delta_normalizer

    def forward(self, observations, actions):
        if self.obs_normalizer:
            observations = self.obs_normalizer.normalize(observations)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        obs_delta_predicted = super().forward(observations, actions)
        if self.delta_normalizer:
            obs_delta_predicted = self.delta_normalizer.denormalize(
                obs_delta_predicted
            )
        return obs_delta_predicted
