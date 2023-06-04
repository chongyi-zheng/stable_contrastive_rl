from rlkit.core import logger
from rlkit.core.timer import timer
from rlkit.data_management.online_vae_replay_buffer import \
    OnlineVaeRelabelingBuffer
from rlkit.data_management.shared_obs_dict_replay_buffer \
    import SharedObsDictRelabelingBuffer
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
)
import rlkit.torch.pytorch_util as ptu
from torch.multiprocessing import Process, Pipe
from threading import Thread
import numpy as np
from rlkit.core.logging import add_prefix

class ActiveRepresentationLearningAlgorithm(TorchBatchRLAlgorithm):

    def __init__(
            self,
            model,
            model_trainer,
            *base_args,
            vae_save_period=1,
            model_training_schedule=vae_schedules.never_train,
            oracle_data=False,
            model_min_num_steps_before_training=0,
            uniform_dataset=None,
            **base_kwargs
    ):
        super().__init__(*base_args, **base_kwargs)
        self.model = model
        self.model_trainer = model_trainer
        self.model_trainer.model = self.model
        self.vae_save_period = vae_save_period
        self.model_training_schedule = model_training_schedule
        self.oracle_data = oracle_data

        self.model_min_num_steps_before_training = model_min_num_steps_before_training
        self.uniform_dataset = uniform_dataset

    def _end_epoch(self):
        timer.start_timer('vae training')
        self._train_vae(self.epoch)
        timer.stop_timer('vae training')
        super()._end_epoch()

    def _get_diagnostics(self):
        vae_log = self._get_vae_diagnostics().copy()
        vae_log.update(super()._get_diagnostics())
        return vae_log

    def to(self, device):
        self.model.to(device)
        super().to(device)

    """
    VAE-specific Code
    """
    def _train_vae(self, epoch):
        should_train, amount_to_train = self.model_training_schedule(epoch)
        rl_start_epoch = int(self.min_num_steps_before_training / (
                self.num_expl_steps_per_train_loop * self.num_train_loops_per_epoch
        ))
        if should_train or epoch <= (rl_start_epoch - 1):
            _train_vae(
                self.model_trainer,
                self.replay_buffer,
                epoch,
                amount_to_train
            )
            # self.replay_buffer.refresh_latents(epoch)
            _test_vae(
                self.model_trainer,
                epoch,
                self.replay_buffer,
                vae_save_period=self.vae_save_period,
                uniform_dataset=self.uniform_dataset,
            )

    def _get_vae_diagnostics(self):
        return add_prefix(
            self.model_trainer.get_diagnostics(),
            prefix='vae_trainer/',
        )


def _train_vae(vae_trainer, replay_buffer, epoch, batches=50, oracle_data=False):
    if oracle_data:
        batch_sampler = None
    vae_trainer.train_epoch(
        epoch,
        replay_buffer,
        batches=batches,
        from_rl=True,
    )


def _test_vae(vae_trainer, epoch, replay_buffer, vae_save_period=1, uniform_dataset=None):
    save_imgs = epoch % vae_save_period == 0
    # log_fit_skew_stats = replay_buffer._prioritize_vae_samples and uniform_dataset is not None
    # if uniform_dataset is not None:
    #     replay_buffer.log_loss_under_uniform(uniform_dataset, vae_trainer.batch_size, rl_logger=vae_trainer.vae_logger_stats_for_rl)
    vae_trainer.test_epoch(
        epoch,
        replay_buffer,
        from_rl=True,
        save_reconstruction=save_imgs,
    )
    # if save_imgs:
    #     vae_trainer.dump_samples(epoch)
    #     if log_fit_skew_stats:
    #         replay_buffer.dump_best_reconstruction(epoch)
    #         replay_buffer.dump_worst_reconstruction(epoch)
    #         replay_buffer.dump_sampling_histogram(epoch, batch_size=vae_trainer.batch_size)
    #     if uniform_dataset is not None:
    #         replay_buffer.dump_uniform_imgs_and_reconstructions(dataset=uniform_dataset, epoch=epoch)
