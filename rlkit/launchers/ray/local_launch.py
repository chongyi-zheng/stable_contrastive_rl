import cloudpickle
import logging

import ray
import ray.tune as tune
from ray.tune.logger import JsonLogger

from rlkit.core.ray_experiment import SequentialRayExperiment
from rlkit.core.ray_csv_logger import SequentialCSVLogger
import rlkit.launchers.ray_config as config

def launch_local_experiment(init_algo_functions_and_log_fnames,
                            exp_variant, use_gpu=False,
                            exp_prefix='test', seeds=1, checkpoint_freq=50,
                            max_failures=10, resume=False, local_ray=True,
                            from_remote=False, resources_per_trial=None,
                            logging_level=logging.DEBUG):
    """Launches a ray experiment locally

    Args:
        init_algo_functions_and_log_fnames ((function, str)[]): a list of tuples.
            The first element of each tuple is a function that returns an algo
            in ray format (i.e, has a _train() method that returns a log dict
            and will train for a single epoch). The second element is the
            filename of the logging file.
        exp_variant (dict): the experiment variant. This will be passed in each
            time to each init_algo_function in init_algo_functions_and_log_fnames
        use_gpu (bool):
        exp_prefix (str):
        seeds (int):
        checkpoint_freq (int): how often to checkpoint for handling failures.
        max_failures (int): how many times to retry if a trial fails. Useful for
            remote launching.
        resume (bool): whether the trials should try and resume a failed trial
            if possible.
        local_ray (bool): whether to use local_ray mode. stdout get printed and
            pdb is possible in local_ray=True.
        from_remote (bool): If the experiment is being launched from a remote
            instance. User should not set this. Automatically set by
            remote_launch.py
        resources_per_trial (dict): Specify {'cpu': float, 'gpu': float}. This
            is the number of allocated resources to the trial.
        logging_level:
    """
    if from_remote:
        redis_address = ray.services.get_node_ip_address() + ':6379'
        ray.init(redis_address=redis_address, logging_level=logging_level)
    else:
        ray.init(local_mode=local_ray)
    for idx, (init_func, log_fname) in enumerate(init_algo_functions_and_log_fnames):
        init_algo_functions_and_log_fnames[idx] = (
            tune.function(init_func),
            log_fname
        )
    exp = tune.Experiment(
        name=exp_prefix,
        run=SequentialRayExperiment,
        upload_dir=config.LOG_BUCKET,
        num_samples=seeds,
        stop={"global_done": True},
        config={
            'algo_variant': exp_variant,
            'init_algo_functions_and_log_fnames': init_algo_functions_and_log_fnames,
            'use_gpu': use_gpu,
            'resources_per_trial': resources_per_trial,
        },
        resources_per_trial=resources_per_trial,
        checkpoint_freq=checkpoint_freq,
        loggers=[JsonLogger, SequentialCSVLogger],
    )
    tune.run(
        exp,
        resume=resume,
        max_failures=max_failures,
        queue_trials=True,
    )


"""
This main should only be invoked by the ray on the remote instance. See
remote_launch.py. The experiment info is pickled and uploaded to the remote
instance. Then, ray invokes this file to start a local experiment from the pkl.
"""
if __name__ == "__main__":
    with open(config.EXPERIMENT_INFO_PKL_FILEPATH, "rb") as f:
        local_launch_variant = cloudpickle.load(f)
    launch_local_experiment(**local_launch_variant)
