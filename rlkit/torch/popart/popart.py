"""
See https://arxiv.org/pdf/1602.07714.pdf
Usage:

Before:
```
net = create_network()

x, y = sample()
y_hat = net(x)
loss = (y - y_hat)**2
```

After:
```
net = create_network()
popart_params = create_popart_params(y_shape)

x, y = sample()
popart_params = popart.update(popart_params, beta, y)
raw_y_hat = net(x)
normalized_y_hat = popart.compute_normalized_prediction(raw_y_hat)
normalized_y = popart.normalize_target(y, popart_params)
loss = (normalized_y - normalized_y_hat)**2

y_hat = popart.compute_prediction(raw_y_hat, popart_params)  # if needed
```
"""
from collections import namedtuple
from numbers import Number
from typing import Union

import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu

Output = Union[torch.Tensor, np.ndarray, Number]
PopArtParams = namedtuple(
    'PopArtParams',
    'mean stddev second_moment w b t beta min_stddev'
)


def create_popart_params(
        output_shape,
        beta=0.001,
        min_stddev=1e-4,
        torch=False,
):
    if torch:
        array_init = ptu
    else:
        array_init = np
    return PopArtParams(
        mean=array_init.zeros(output_shape),
        stddev=array_init.ones(output_shape),
        second_moment=array_init.ones(output_shape),
        w=array_init.ones(output_shape),
        b=array_init.zeros(output_shape),
        t=1,
        beta=beta,
        min_stddev=min_stddev
    )


def update(
        popart_params: PopArtParams,
        target: Output,
) -> PopArtParams:
    """
    Update popart params based on some sampled target.

    :param popart_params:
    :param target:
    :return:
    """
    old_mean, old_stddev = popart_params.mean, popart_params.stddev
    old_second_moment = popart_params.second_moment
    old_w, old_b = popart_params.w, popart_params.b
    t = popart_params.t

    sample_first = target.mean()
    sample_second = (target ** 2).mean()
    beta = popart_params.beta / (1 - (1-popart_params.beta)**t)
    new_mean = old_mean * (1-beta) + sample_first * beta
    new_second_moment = old_second_moment * (1-beta) + sample_second * beta
    if isinstance(new_second_moment, torch.Tensor):
        new_stddev = (new_second_moment - new_mean ** 2).sqrt()
        new_stddev = new_stddev.clamp(min=popart_params.min_stddev)
    else:
        new_stddev = np.sqrt(new_second_moment - new_mean ** 2)
        new_stddev = np.clip(new_stddev, popart_params.min_stddev, None)

    new_w = old_w * old_stddev / new_stddev
    new_b = (old_stddev * old_b + old_mean - new_mean) / new_stddev
    return PopArtParams(
        new_mean,
        new_stddev,
        new_second_moment,
        new_w,
        new_b,
        t+1,
        popart_params.beta,
        popart_params.min_stddev,
    )


def normalize_target(y: Output, popart_params: PopArtParams):
    mean, stddev = popart_params.mean, popart_params.stddev
    return (y - mean) / stddev


def unnormalize_prediction(y: Output, popart_params: PopArtParams):
    mean, stddev = popart_params.mean, popart_params.stddev
    return y * stddev + mean


def compute_normalized_prediction(y: Output, popart_params: PopArtParams):
    w, b = popart_params.w, popart_params.b
    return w * y + b


def compute_prediction(y: Output, popart_params: PopArtParams):
    mean, std = popart_params.mean, popart_params.stddev
    w, b = popart_params.w, popart_params.b
    return (w * y + b) * std + mean


def interpolate(first: PopArtParams, second: PopArtParams, first_weight: float):
    if first_weight > 1 or first_weight < 0:
        raise ValueError("weight must be in range [0, 1]")
    new_params = tuple(
        a * first_weight + b * (1-first_weight)
        for a, b in zip(first, second)
    )
    return PopArtParams(*new_params)
