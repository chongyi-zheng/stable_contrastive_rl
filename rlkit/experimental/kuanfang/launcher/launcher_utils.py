import os
import time
from typing import NamedTuple
import random

import __main__ as main
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger, setup_logger
from rlkit.launchers import config
import torch


def run_experiment(
        method_call,
        exp_name='default',
        mode='local',
        variant=None,
        use_gpu=False,
        gpu_id=0,
        wrap_fn_with_auto_setup=True,
        unpack_variant=True,
        base_log_dir=None,
        prepend_date_to_exp_name=True,
        slurm_config=None,
        **kwargs
):
    if base_log_dir is None:
        base_log_dir = config.LOCAL_LOG_DIR
    if wrap_fn_with_auto_setup:
        method_call = AutoSetup(method_call, unpack_variant=unpack_variant)
    if mode == 'here_no_doodad':
        if prepend_date_to_exp_name:
            exp_name = time.strftime("%y-%m-%d") + "-" + exp_name
        setup_experiment(
            variant=variant,
            exp_name=exp_name,
            base_log_dir=base_log_dir,
            git_infos=generate_git_infos(),
            script_name=main.__file__,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
        )
        method_call(None, variant)
    else:
        from doodad.easy_launch.python_function import (
            run_experiment as doodad_run_experiment
        )
        doodad_run_experiment(
            method_call,
            exp_name=exp_name,
            mode=mode,
            variant=variant,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            prepend_date_to_exp_name=prepend_date_to_exp_name,
            **kwargs
        )


def setup_experiment(
        variant,
        exp_name,
        base_log_dir,
        git_infos,
        script_name,
        use_gpu,
        gpu_id,
):
    logger_config = variant.get('logger_config', {})
    seed = variant.get('seed', random.randint(0, 999999))
    exp_id = variant.get('exp_id', random.randint(0, 999999))
    set_seed(seed)
    ptu.set_gpu_mode(use_gpu, gpu_id)
    os.environ['gpu_id'] = str(gpu_id)
    setup_logger(
        logger,
        exp_name=exp_name,
        base_log_dir=base_log_dir,
        variant=variant,
        git_infos=git_infos,
        script_name=script_name,
        seed=seed,
        exp_id=exp_id,
        **logger_config)

    print('The snapshot_dir of the logger is: %s' % (
        logger.get_snapshot_dir()))


def run_variant(experiment, variant):
    launcher_config = variant.get("launcher_config")
    lu.run_experiment(
        experiment,
        variant=variant,
        **launcher_config,
    )
