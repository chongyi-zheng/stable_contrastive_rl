"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from rlkit.torch.networks.basic import (
    Sigmoid, Clamp, SigmoidClamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from rlkit.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy, TwoChannelCNN, ConcatTwoChannelCNN
from rlkit.torch.networks.dcnn import DCNN, TwoHeadDCNN
from rlkit.torch.networks.deprecated_feedforward import (
    FeedForwardPolicy, FeedForwardQFunction
)
from rlkit.torch.networks.feat_point_mlp import FeatPointMlp
from rlkit.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from rlkit.torch.networks.linear_transform import LinearTransform
from rlkit.torch.networks.mlp import (
    Mlp, ConcatMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
)
from rlkit.torch.networks.pretrained_cnn import PretrainedCNN
from rlkit.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Sigmoid',
    'Clamp',
    'SigmoidClamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'TwoChannelCNN',
    "ConcatTwoChannelCNN",
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeedForwardPolicy',
    'FeedForwardQFunction',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

