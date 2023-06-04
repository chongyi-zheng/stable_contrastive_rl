import abc
from typing import Dict
from gym import Space


class DictDistribution(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def sample(self, batch_size: int):
        pass

    @property
    @abc.abstractmethod
    def spaces(self) -> Dict[str, Space]:
        pass

    def __call__(self, *args, **kwargs):
        """For backward compatibility with DictDistributionGenerator"""
        return self


class DictDistributionGenerator(DictDistribution, metaclass=abc.ABCMeta):

    def __call__(self, *args, **kwargs) -> DictDistribution:
        raise NotImplementedError


class DictDistributionClosure(DictDistributionGenerator):
    """Fills in args to a DictDistribution"""

    def __init__(self, clz, *args, **kwargs):
        self.clz = clz
        self.args = args
        self.kwargs = kwargs

    def __call__(self, **extra_kwargs) -> DictDistribution:
        self.kwargs.update(**extra_kwargs)
        return self.clz(
            *args,
            **kwargs,
        )
