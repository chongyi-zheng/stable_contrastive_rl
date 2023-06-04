from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
from torch import optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.utils import save_image
from rlkit.data_management.images import normalize_image
from rlkit.core import logger
import rlkit.core.util as util
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.ml_util import ConstantSchedule
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.data import (
    ImageDataset, InfiniteWeightedRandomSampler,
    InfiniteRandomSampler,
)
from rlkit.torch.core import np_to_pytorch_batch
import collections
from rlkit.torch.vae.vae_trainer import ConvVAETrainer

class ConditionalConvVAETrainer(ConvVAETrainer):
    def compute_loss(self, batch, epoch, test=False):
        prefix = "test/" if test else "train/"

        beta = float(self.beta_schedule.get_value(epoch))
        obs = batch["observations"]
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(obs)
        log_prob = self.model.logprob(batch["x_t"], obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        loss = -1 * log_prob + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (batch, reconstructions)

        if test:
            self.test_last_batch = (obs, reconstructions)

        return loss


    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        x0 = batch["x_0"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            x0[:n].narrow(start=0, length=self.imlength // 2, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength // 2, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_samples(self, epoch):
        self.model.eval()
        batch, _ = self.eval_data["test/last_batch"]
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample, batch["observations"])[0].cpu()
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(
            sample.data.view(64, 3, self.imsize, self.imsize).transpose(2, 3),
            save_dir
        )

        x0 = batch["x_0"]
        x0_img = x0[:64].narrow(start=0, length=self.imlength // 2, dim=1).contiguous().view(
            -1,
            3,
            self.imsize,
            self.imsize
        ).transpose(2, 3)
        save_dir = osp.join(self.log_dir, 'x0_%d.png' % epoch)
        save_image(x0_img.data.cpu(), save_dir)


class CVAETrainer(ConditionalConvVAETrainer):

    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0.001,
            key_to_reconstruct='x_t',
            num_epochs=500,
    ):
        super().__init__(
            model,
            batch_size,
            log_interval,
            beta,
            beta_schedule,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            linearity_weight,
            distance_weight,
            loss_weights,
            use_linear_dynamics,
            use_parallel_dataloading,
            train_data_workers,
            skew_dataset,
            skew_config,
            priority_function_kwargs,
            start_skew_epoch,
            weight_decay,
            key_to_reconstruct,
            num_epochs
        )
        self.optimizer = optim.Adam(self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay,
        )

    def compute_loss(self, batch, epoch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(batch["x_t"], batch["env"])
        log_prob = self.model.logprob(batch["x_t"], obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        loss = -1 * log_prob + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (batch, reconstructions)

        return loss

    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3),
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_distances(self, batch, epoch):
        import matplotlib.pyplot as plt
        plt.clf()
        state = batch['episode_obs']
        size = self.model.imsize
        n = min(state.size(0), 8)
        if n <= 2: return
        distances = []
        all_imgs = [state[i].reshape(3, size, size).transpose(1, 2) for i in range(n)]
        env, goal = state[0].reshape(1,-1), state[-1].reshape(1,-1)
        latent_goal = self.model.encode(goal, env, distrib=False)

        for i in range(n):
            latent = self.model.encode(state[i].reshape(1,-1), env, distrib=False)
            distances.append(np.linalg.norm(ptu.get_numpy(latent) - ptu.get_numpy(latent_goal)))

        plt.plot(np.arange(n), np.array(distances))
        save_dir = osp.join(self.log_dir, 'dist_%d_plot.png' % epoch)
        plt.savefig(save_dir)

        all_imgs = torch.stack(all_imgs)
        save_dir = osp.join(self.log_dir, 'dist_%d_traj.png' % epoch)
        save_image(
            all_imgs.data,
            save_dir,
            nrow=n,
        )

    def dump_samples(self, epoch):
        self.model.eval()
        batch, reconstructions, = self.eval_data["test/last_batch"]
        self.dump_distances(batch, epoch)
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            latent = self.model.sample_prior(self.batch_size, env)
            samples = self.model.decode(latent)[0]
            all_imgs.extend([
                samples.view(
                    self.batch_size,
                    3,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=8)


class DeltaCVAETrainer(ConditionalConvVAETrainer):

    def __init__(
            self,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            context_schedule=None,
            lr=None,
            do_scatterplot=False,
            normalize=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            background_subtract=False,
            linearity_weight=0.0,
            distance_weight=0.0,
            loss_weights=None,
            use_linear_dynamics=False,
            use_parallel_dataloading=False,
            train_data_workers=2,
            skew_dataset=False,
            skew_config=None,
            priority_function_kwargs=None,
            start_skew_epoch=0,
            weight_decay=0.001,
            num_epochs=500,
    ):
        super().__init__(
            model,
            batch_size,
            log_interval,
            beta,
            beta_schedule,
            lr,
            do_scatterplot,
            normalize,
            mse_weight,
            is_auto_encoder,
            background_subtract,
            linearity_weight,
            distance_weight,
            loss_weights,
            use_linear_dynamics,
            use_parallel_dataloading,
            train_data_workers,
            skew_dataset,
            skew_config,
            priority_function_kwargs,
            start_skew_epoch,
            weight_decay,
        )
        self.context_schedule = context_schedule
        self.optimizer = optim.Adam(self.model.parameters(),
            lr=self.lr,
            weight_decay=weight_decay,
        )

    def compute_loss(self, batch, epoch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        #context_weight = float(self.context_schedule.get_value(epoch))
        x_t, env = self.model(batch["x_t"], batch["env"])
        reconstructions, obs_distribution_params, latent_distribution_params = x_t
        env_reconstructions, env_distribution_params = env
        log_prob = self.model.logprob(batch["x_t"], obs_distribution_params)
        env_log_prob = self.model.logprob(batch["env"], env_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        loss = -1 * (log_prob + env_log_prob) + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "env_log_probs"].append(env_log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (batch, reconstructions, env_reconstructions)

        return loss

    def dump_mixed_latents(self, epoch):
        n = 8
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        x_t, env = batch["x_t"][:n], batch["env"][:n]
        z_pos, logvar, z_obj = self.model.encode(x_t, env)

        grid = []
        for i in range(n):
            for j in range(n):
                if i + j == 0:
                    grid.append(ptu.zeros(1, self.input_channels, self.imsize, self.imsize))
                elif i == 0:
                    grid.append(x_t[j].reshape(1, self.input_channels, self.imsize, self.imsize))
                elif j == 0:
                    grid.append(env[i].reshape(1, self.input_channels, self.imsize, self.imsize))
                else:
                    pos, obj = z_pos[j].reshape(1, -1), z_obj[i].reshape(1, -1)
                    img = self.model.decode(torch.cat([pos, obj], dim=1))[0]
                    grid.append(img.reshape(1, self.input_channels, self.imsize, self.imsize))
        samples = torch.cat(grid)
        save_dir = osp.join(self.log_dir, 'mixed_latents_%d.png' % epoch)
        save_image(samples.data.cpu().transpose(2, 3), save_dir, nrow=n)

    def dump_reconstructions(self, epoch):
        self.dump_mixed_latents(epoch)

        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3),
            env_reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_samples(self, epoch):
        self.model.eval()
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        # self.dump_distances(batch, epoch)
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            latent = self.model.sample_prior(self.batch_size, env)
            samples = self.model.decode(latent)[0]
            all_imgs.extend([
                samples.view(
                    self.batch_size,
                    3,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=8)


class CDVAETrainer(CVAETrainer):

    def state_linearity_loss(self, x_t, x_next, env, actions):
        latent_obs = self.model.encode(x_t, env, distrib=False)
        latent_next_obs = self.model.encode(x_next, env, distrib=False)
        predicted_latent = self.model.process_dynamics(latent_obs, actions)
        return torch.norm(predicted_latent - latent_next_obs) ** 2 / self.batch_size

    def state_distance_loss(self, x_t, x_next, env):
        latent_obs = self.model.encode(x_t, env, distrib=False)
        latent_next_obs = self.model.encode(x_next, env, distrib=False)
        return torch.norm(latent_obs - latent_next_obs) ** 2 / self.batch_size

    def compute_loss(self, batch, epoch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        reconstructions, obs_distribution_params, latent_distribution_params = self.model(batch["x_t"], batch["env"])
        log_prob = self.model.logprob(batch["x_t"], obs_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        state_distance_loss = self.state_distance_loss(batch["x_t"], batch["x_next"], batch["env"])
        dynamics_loss = self.state_linearity_loss(
            batch["x_t"], batch["x_next"], batch["env"], batch["actions"]
        )

        loss = -1 * log_prob + beta * kle + self.linearity_weight * dynamics_loss + self.distance_weight * state_distance_loss

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())
        self.eval_statistics[prefix + "dynamics_loss"].append(dynamics_loss.item())
        self.eval_statistics[prefix + "distance_loss"].append(state_distance_loss.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (batch, reconstructions)

        return loss

    def dump_dynamics(self, batch, epoch):
        self.model.eval()
        state = batch['episode_obs']
        act = batch['episode_acts']
        size = self.model.imsize
        n = min(state.size(0), 8)

        all_imgs = [state[i].reshape(3, size, size).transpose(1, 2) for i in range(n)]
        latent_state = self.model.encode(state[0].reshape(1, -1), state[0].reshape(1, -1), distrib=False)
        pred_curr = self.model.decode(latent_state)[0]
        all_imgs.append(pred_curr.view(3, size, size).transpose(1, 2))

        for i in range(n - 1):
            latent_state = self.model.process_dynamics(latent_state.reshape(1, -1), act[i].reshape(1, -1))
            pred_curr = self.model.decode(latent_state)[0]
            all_imgs.append(pred_curr.view(3, size, size).transpose(1, 2))

        all_imgs = torch.stack(all_imgs)
        save_dir = osp.join(self.log_dir, 'dynamics%d.png' % epoch)
        save_image(
            all_imgs.data,
            save_dir,
            nrow=n,
        )

    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
        self.dump_dynamics(batch, epoch)
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

class DeltaDynamicsCVAETrainer(CDVAETrainer):

    def compute_loss(self, batch, epoch, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        x_t, env = self.model(batch["x_t"], batch["env"])
        reconstructions, obs_distribution_params, latent_distribution_params = x_t
        env_reconstructions, env_distribution_params = env
        log_prob = self.model.logprob(batch["x_t"], obs_distribution_params)
        env_log_prob = self.model.logprob(batch["env"], env_distribution_params)
        kle = self.model.kl_divergence(latent_distribution_params)
        state_distance_loss = self.state_distance_loss(batch["x_t"], batch["x_next"], batch["env"])
        dynamics_loss = self.state_linearity_loss(
            batch["x_t"], batch["x_next"], batch["env"], batch["actions"]
        )

        loss = -1 * (log_prob + env_log_prob) + beta * kle + self.linearity_weight * dynamics_loss + self.distance_weight * state_distance_loss

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "log_probs"].append(log_prob.item())
        self.eval_statistics[prefix + "env_log_probs"].append(env_log_prob.item())
        self.eval_statistics[prefix + "kles"].append(kle.item())
        self.eval_statistics[prefix + "dynamics_loss"].append(dynamics_loss.item())
        self.eval_statistics[prefix + "distance_loss"].append(state_distance_loss.item())

        encoder_mean = self.model.get_encoding_from_latent_distribution_params(latent_distribution_params)
        z_data = ptu.get_numpy(encoder_mean.cpu())
        for i in range(len(z_data)):
            self.eval_data[prefix + "zs"].append(z_data[i, :])
        self.eval_data[prefix + "last_batch"] = (batch, reconstructions, env_reconstructions)

        return loss

    def dump_samples(self, epoch):
        self.model.eval()
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        self.dump_distances(batch, epoch)
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            latent = self.model.sample_prior(self.batch_size, env)
            samples = self.model.decode(latent)[0]
            all_imgs.extend([
                samples.view(
                    self.batch_size,
                    3,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=8)


    def dump_reconstructions(self, epoch):
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        self.dump_dynamics(batch, epoch)
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                3,
                self.imsize,
                self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3),
            env_reconstructions.view(
                self.batch_size,
                3,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)
