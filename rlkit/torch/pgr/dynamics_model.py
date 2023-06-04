from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class EnsembleToGaussian(DistributionGenerator):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble

    def forward(self, *args, **kwargs):
        predictions = self.ensemble(*args, **kwargs)
        mean = predictions.mean(dim=-1)
        std = predictions.std(dim=-1)
        return MultivariateDiagonalNormal(mean, std)
