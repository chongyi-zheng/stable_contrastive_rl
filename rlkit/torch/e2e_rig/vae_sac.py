from collections import OrderedDict

from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.vae.vae_trainer import VAETrainer
import rlkit.torch.pytorch_util as ptu

import torch.optim as optim
import itertools



class End2EndSACTrainer(TorchTrainer):
    def __init__(
            self,
            sac_trainer: SACTrainer,
            vae_trainer: VAETrainer,
            combined_lr=1e-4,
            vae_training_method='separate',
    ):
        """

        :param sac_trainer:
        :param vae_trainer:
        :param combined_lr:
        :param vae_training_method:
            'vae': use standard vae training loss.
            'none': do not train the VAE at all (fixed initialization.
            'vae_and_qf': use the VAE loss and Bellman error to train the VAE.
        """
        super().__init__()
        self._sac_trainer = sac_trainer
        self._vae_trainer = vae_trainer
        self._need_to_update_eval_statistics = True
        self._vae_training_method = vae_training_method

        self.eval_statistics = OrderedDict()

        # TODO: Consider refactoring this.
        # Seems like we might want to isolate this trainer more from
        # sac_trainer/vae_trainer
        self.alpha_optimizer = self._sac_trainer.alpha_optimizer
        self.qf1_optimizer = self._sac_trainer.qf1_optimizer
        self.qf2_optimizer = self._sac_trainer.qf2_optimizer
        self.policy_optimizer = self._sac_trainer.policy_optimizer
        self.target_update_period = self._sac_trainer.target_update_period

        if self._vae_training_method == 'vae':
            self.vae_optimizer = self._vae_trainer.optimizer
        elif self._vae_training_method == 'none':
            pass
        elif self._vae_training_method == 'vae_and_qf':
            qf_and_vae_params = itertools.chain(
                self._vae_trainer.model.parameters(),
                self._sac_trainer.qf1.parameters(),
                self._sac_trainer.qf2.parameters(),
            )
            self.combined_optimizer = optim.Adam(
                qf_and_vae_params,
                lr=combined_lr,
            )
        else:
            raise ValueError('Unknown vae training method: {}'.format(
                vae_training_method))
        self._n_train_steps_total = 0

    def train_from_torch(self, batch):
        losses = self._sac_trainer.compute_loss(
            batch,
            update_eval_statistics=self._need_to_update_eval_statistics,
        )
        vae_loss = self._vae_trainer.compute_loss(batch, test=False)

        """
        Update networks
        """
        self.alpha_optimizer.zero_grad()
        losses.alpha_loss.backward()
        self.alpha_optimizer.step()

        if self._vae_training_method == 'vae':
            self.qf1_optimizer.zero_grad()
            losses.qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            losses.qf2_loss.backward()
            self.qf2_optimizer.step()

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
        elif self._vae_training_method == 'none':
            self.qf1_optimizer.zero_grad()
            losses.qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            losses.qf2_loss.backward()
            self.qf2_optimizer.step()
        elif self._vae_training_method == 'vae_and_qf':
            combined_loss = losses.qf2_loss + losses.qf1_loss + vae_loss
            self.combined_optimizer.zero_grad()
            combined_loss.backward()
            self.combined_optimizer.step()
        else:
            raise ValueError('Unknown vae training method: {}'.format(
                self._vae_training_method))

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            self._sac_trainer.update_target_networks()
        self._n_train_steps_total += 1

        if self._need_to_update_eval_statistics:
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
            self.eval_statistics['vae_loss'] = ptu.get_numpy(vae_loss)
            self.eval_statistics['qf1_loss'] = ptu.get_numpy(losses.qf1_loss)
            self.eval_statistics['qf2_loss'] = ptu.get_numpy(losses.qf2_loss)

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self._sac_trainer.get_diagnostics())
        stats.update(self._vae_trainer.get_diagnostics())
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return self._sac_trainer.networks + [self._vae_trainer.model]

    def get_snapshot(self):
        return self._sac_trainer.get_snapshot()
