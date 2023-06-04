import abc

from torch import nn

from rlkit.torch.distributions import (
    Bernoulli,
    Beta,
    Distribution,
    Independent,
    GaussianMixture as GaussianMixtureDistribution,
    GaussianMixtureFull as GaussianMixtureFullDistribution,
    MultivariateDiagonalNormal,
    TanhNormal,
    IndependentLaplace,
)
from rlkit.torch.networks.basic import MultiInputSequential


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError


class ModuleToDistributionGenerator(
    MultiInputSequential,
    DistributionGenerator,
    metaclass=abc.ABCMeta
):
    pass


class Beta(ModuleToDistributionGenerator):
    def forward(self, *input):
        alpha, beta = super().forward(*input)
        return Beta(alpha, beta, validate_args=False)


class Gaussian(ModuleToDistributionGenerator):
    def __init__(self, module, std=None, reinterpreted_batch_ndims=1):
        super().__init__(module)
        self.std = std
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        if self.std:
            mean = super().forward(*input)
            std = self.std
        else:
            mean, log_std = super().forward(*input)
            std = log_std.exp()
        return MultivariateDiagonalNormal(
            mean, std, reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)


class BernoulliGenerator(ModuleToDistributionGenerator):
    def forward(self, *input):
        probs = super().forward(*input)
        return Bernoulli(probs, validate_args=False)


class IndependentGenerator(ModuleToDistributionGenerator):
    def __init__(self, *args, reinterpreted_batch_ndims=1):
        super().__init__(*args)
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def forward(self, *input):
        distribution = super().forward(*input)
        return Independent(
            distribution,
            reinterpreted_batch_ndims=self.reinterpreted_batch_ndims,
            validate_args=False
        )


class IndependentLaplaceGen(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return IndependentLaplace(mean, std)


class GaussianMixture(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureDistribution(mixture_means, mixture_stds, weights)


class GaussianMixtureFull(ModuleToDistributionGenerator):
    def forward(self, *input):
        mixture_means, mixture_stds, weights = super().forward(*input)
        return GaussianMixtureFullDistribution(mixture_means, mixture_stds, weights)


class TanhGaussian(ModuleToDistributionGenerator):
    def forward(self, *input):
        mean, log_std = super().forward(*input)
        std = log_std.exp()
        return TanhNormal(mean, std)
