import os
import time
from typing import NamedTuple
import random

import __main__ as main
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, setup_logger
from rlkit.launchers import config

from doodad.wrappers.easy_launch import save_doodad_config, sweep_function, DoodadConfig
from rlkit.launchers.launcher_util import set_seed


class AutoSetup:
    """
    Automatically set up:
    1. the logger
    2. the GPU mode
    3. the seed
    :param exp_function: some function that should not depend on `logger_config`
    nor `seed`.
    :param unpack_variant: do you call exp_function with `**variant`?
    :return: function output
    """
    def __init__(self, exp_function, unpack_variant=True):
        self.exp_function = exp_function
        self.unpack_variant = unpack_variant

    def __call__(self, doodad_config: DoodadConfig, variant):
        save_doodad_config(doodad_config)
        variant_to_save = variant.copy()
        variant_to_save['doodad_info'] = doodad_config.extra_launch_info
        exp_name = doodad_config.extra_launch_info['exp_name']
        seed = variant.pop('seed', 0)
        set_seed(seed)
        ptu.set_gpu_mode(doodad_config.use_gpu)
        # Reopening the files is nececessary because blobfuse only syncs files
        # when they're closed. For details, see
        # https://github.com/Azure/azure-storage-fuse#if-your-workload-is-not-read-only
        reopen_files_on_flush = True
        # might as well always have it on, but if I didn't want to, you could:
        # mode = doodad_config.extra_launch_info['mode']
        # reopen_files_on_flush = mode == 'azure'
        setup_logger(
            logger,
            exp_name=exp_name,
            base_log_dir=None,
            log_dir=doodad_config.output_directory,
            seed=seed,
            variant=variant,
            reopen_files_on_flush=reopen_files_on_flush,
        )
        variant.pop('logger_config', None)
        variant.pop('exp_id', None)
        variant.pop('run_id', None)
        if self.unpack_variant:
            self.exp_function(**variant)
        else:
            self.exp_function(variant)


def run_experiment(
        method_call,
        params,
        default_params,
        exp_name='default',
        mode='local',
        wrap_fn_with_auto_setup=True,
        unpack_variant=True,
        **kwargs
):
    if wrap_fn_with_auto_setup:
        method_call = AutoSetup(
            method_call,
            unpack_variant=unpack_variant,
        )
    sweep_function(
        method_call,
        params,
        default_params=default_params,
        mode=mode,
        log_path=exp_name,
        add_time_to_run_id='in_front',
        extra_launch_info={'exp_name': exp_name, 'mode': mode},
        **kwargs
    )
