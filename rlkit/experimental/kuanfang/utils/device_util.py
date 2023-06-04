from absl import logging  # NOQA

import torch

import rlkit.torch.pytorch_util as ptu


def set_device(use_gpu, gpu_id=None):
    if use_gpu:
        assert torch.cuda.is_available()

        if gpu_id is None:
            gpu_id = 0

    else:
        gpu_id = -1

    ptu.set_gpu_mode(mode=use_gpu, gpu_id=gpu_id)
    logging.info('Device: %r', ptu.device)
