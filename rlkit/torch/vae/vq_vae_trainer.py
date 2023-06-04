from collections import OrderedDict
import os
from os import path as osp
import numpy as np
import torch
from rlkit.core.loss import LossFunction
from rlkit.torch.vae.vae_trainer import ConvVAETrainer
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
import time


class VQ_VAETrainer(ConvVAETrainer, LossFunction):

    def train_batch(self, epoch, batch):
        self.num_batches += 1
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch, epoch, False)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test_batch(
            self,
            epoch,
            batch,
    ):
        self.model.eval()
        loss = self.compute_loss(batch, epoch, True)

    def encode_dataset(self, dataset):
        encoding_list = []
        save_dir = osp.join(self.log_dir, 'dataset_latents.npy')
        for i in range(len(dataset)):
            batch = dataset.random_batch(self.batch_size)
            obs, cond = batch["x_t"], batch["env"]
            z_delta = self.model.encode(obs, cont=False)
            z_cond = self.model.encode(cond, cont=False)
            encodings = torch.cat([z_delta, z_cond], dim=1)
            encoding_list.append(encodings)
        encodings = ptu.get_numpy(torch.cat(encoding_list))
        np.save(save_dir, encodings)

    def train_epoch(self, epoch, dataset, batches=100):
        start_time = time.time()
        for b in range(batches):
            batch = dataset.random_batch(self.batch_size)

            self.train_batch(epoch, batch)
        self.eval_statistics["train/epoch_duration"].append(time.time() - start_time)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        obs = batch[self.key_to_reconstruct]

        vq_loss, data_recon, perplexity, recon_error = self.model.compute_loss(obs)
        loss = vq_loss + recon_error

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['num_batches'] = self.num_batches
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "Recon Error"].append(recon_error.item())
        self.eval_statistics[prefix + "VQ Loss"].append(vq_loss.item())
        self.eval_statistics[prefix + "Perplexity"].append(perplexity.item())
        self.eval_data[prefix + "last_batch"] = (obs, data_recon.detach())

        return loss

    def dump_reconstructions(self, epoch):
        obs, reconstructions = self.eval_data["test/last_batch"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1, self.input_channels, self.imsize, self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                self.input_channels,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'test_recon_%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

        obs, reconstructions = self.eval_data["train/last_batch"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            obs[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1, self.input_channels, self.imsize, self.imsize
            ).transpose(2, 3),
            reconstructions.view(
                self.batch_size,
                self.input_channels,
                self.imsize,
                self.imsize,
            )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'train_recon_%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)

    def dump_samples(self, epoch):
        return


class VAETrainer(VQ_VAETrainer):

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        recon, error, kle = self.model.compute_loss(batch["x_t"])
        loss = error + beta * kle

        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "kle"].append(kle.item())
        self.eval_statistics[prefix + "Obs Recon Error"].append(error.item())
        self.eval_data[prefix + "last_batch"] = (batch, recon)

        return loss


    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.randn(64, self.representation_size)
        sample = self.model.decode(sample)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(
            sample.data.transpose(2, 3),
            save_dir
        )

    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
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



class CCVAETrainer(VQ_VAETrainer):

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        recon, x_recon_error, c_recon_error, kle = self.model.compute_loss(batch["x_t"], batch["env"])
        loss = x_recon_error + c_recon_error + beta * kle
        self.eval_statistics['epoch'] = epoch
        self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        self.eval_statistics[prefix + "kle"].append(kle.item())
        self.eval_statistics[prefix + "Obs Recon Error"].append(x_recon_error.item())
        self.eval_statistics[prefix + "Cond Obs Recon Error"].append(c_recon_error.item())
        self.eval_data[prefix + "last_batch"] = (batch, recon)

        return loss

    def dump_mixed_latents(self, epoch):
        n = 8
        batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
        x_t, env = batch["x_t"][:n], batch["env"][:n]
        z_comb = self.model.encode(x_t, env)

        z_pos = z_comb[:, :self.model.latent_sizes[0]]
        z_obj = z_comb[:, self.model.latent_sizes[0]:]
        grid = []
        for i in range(n):
            for j in range(n):
                if i + j == 0:
                    grid.append(ptu.zeros(1, self.input_channels, self.imsize, self.imsize))
                elif i == 0:
                    #grid.append(self.model.decode(torch.cat([z_pos[j], z_obj[i]], dim=1)))
                    grid.append(x_t[j].reshape(1, self.input_channels, self.imsize, self.imsize))
                elif j == 0:
                    #grid.append(self.model.decode(torch.cat([z_pos[j], z_obj[i]], dim=1)))
                    grid.append(env[i].reshape(1, self.input_channels, self.imsize, self.imsize))
                else:
                    z, z_c = z_pos[j].reshape(1, -1), z_obj[i].reshape(1, -1)
                    grid.append(self.model.decode(torch.cat([z, z_c], dim=1)))
        samples = torch.cat(grid)
        save_dir = osp.join(self.log_dir, 'mixed_latents_%d.png' % epoch)
        save_image(samples.data.cpu().transpose(2, 3), save_dir, nrow=n)


    def dump_samples(self, epoch):
        self.model.eval()
        batch, reconstructions = self.eval_data["test/last_batch"]
        env = batch["env"]
        n = min(env.size(0), 8)

        all_imgs = [
            env[:n].narrow(start=0, length=self.imlength, dim=1)
                .contiguous().view(
                -1,
                self.input_channels,
                self.imsize,
                self.imsize
            ).transpose(2, 3)]

        for i in range(7):
            latent = ptu.from_numpy(self.model.sample_prior(self.batch_size, ptu.get_numpy(env)))
            samples = self.model.decode(latent)
            all_imgs.extend([
                samples.view(
                    self.batch_size,
                    self.input_channels,
                    self.imsize,
                    self.imsize,
                )[:n].transpose(2, 3)])
        comparison = torch.cat(all_imgs)
        save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=8)


    def dump_reconstructions(self, epoch):
        batch, reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
            # env[:n].narrow(start=0, length=self.imlength, dim=1)
            #     .contiguous().view(
            #     -1,
            #     3,
            #     self.imsize,
            #     self.imsize
            # ).transpose(2, 3),
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
            # env_reconstructions.view(
            #     self.batch_size,
            #     3,
            #     self.imsize,
            #     self.imsize,
            # )[:n].transpose(2, 3)
        ])
        save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
        save_image(comparison.data.cpu(), save_dir, nrow=n)











####### OLD #######
class CVQVAETrainer(VQ_VAETrainer):

    def encode_dataset(self, dataset):
        encoding_list = []
        save_dir = osp.join(self.log_dir, 'dataset_latents.npy')
        for i in range(len(dataset)):
            batch = dataset.random_batch(self.batch_size)
            encodings = self.model.encode(batch["x_t"], batch["env"], cont=False)
            encoding_list.append(encodings)
        encodings = ptu.get_numpy(torch.cat(encoding_list))
        np.save(save_dir, encodings)

    def test_epoch(self, epoch, dataset, batches=10):
        start_time = time.time()
        for b in range(batches):
            self.test_batch(epoch, dataset.random_batch(self.batch_size))
        self.eval_statistics["test/epoch_duration"].append(time.time() - start_time)

    def compute_loss(self, batch, epoch=-1, test=False):
        prefix = "test/" if test else "train/"
        beta = float(self.beta_schedule.get_value(epoch))
        vq_loss, quantized, recon, perplexity, error = self.model.compute_loss(batch["x_t"], batch["env"])
        #vq_loss, perplexity, recon, error = self.model.compute_loss(batch["x_t"], batch["env"])
        loss = error + vq_loss
        #loss = sum(errors) + beta * kle
        self.eval_statistics['epoch'] = epoch
        #self.eval_statistics['beta'] = beta
        self.eval_statistics[prefix + "losses"].append(loss.item())
        #self.eval_statistics[prefix + "kle"].append(kle.item())
        self.eval_statistics[prefix + "Obs Recon Error"].append(error.item())
        # self.eval_statistics[prefix + "Cond Obs Recon Error"].append(errors[1].item())
        self.eval_statistics[prefix + "VQ Loss"].append(vq_loss.item())
        self.eval_statistics[prefix + "Perplexity"].append(perplexity.item())
        # self.eval_statistics[prefix + "Cond VQ Loss"].append(vq_losses[1].item())
        # self.eval_statistics[prefix + "Cond Perplexity"].append(perplexities[1].item())
        self.eval_data[prefix + "last_batch"] = (batch, recon)
        #self.eval_data[prefix + "last_batch"] = (batch, recons[0], recons[1])

        return loss

    def dump_mixed_latents(self, epoch):
        n = 8
        batch, reconstructions = self.eval_data["test/last_batch"]
        x_t, env = batch["x_t"][:n], batch["env"][:n]
        z_comb = self.model.encode(x_t, env)
        z_pos = z_comb[:, :self.model.latent_sizes[0]]
        z_obj = z_comb[:, self.model.latent_sizes[0]:]
        grid = []
        for i in range(n):
            for j in range(n):
                if i + j == 0:
                    grid.append(ptu.zeros(1, self.input_channels, self.imsize, self.imsize))
                elif i == 0:
                    #grid.append(self.model.decode(torch.cat([z_pos[j], z_obj[i]], dim=1)))
                    grid.append(x_t[j].reshape(1, self.input_channels, self.imsize, self.imsize))
                elif j == 0:
                    #grid.append(self.model.decode(torch.cat([z_pos[j], z_obj[i]], dim=1)))
                    grid.append(env[i].reshape(1, self.input_channels, self.imsize, self.imsize))
                else:
                    z, z_c = z_pos[j].reshape(1, -1), z_obj[i].reshape(1, -1)
                    grid.append(self.model.decode(torch.cat([z, z_c], dim=1)))
        samples = torch.cat(grid)
        save_dir = osp.join(self.log_dir, 'mixed_latents_%d.png' % epoch)
        save_image(samples.data.cpu().transpose(2, 3), save_dir, nrow=n)

    def dump_samples(self, epoch):
        return
    # def dump_samples(self, epoch):
    #     self.model.eval()
    #     batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
    #     #self.dump_distances(batch, epoch)
    #     env = batch["env"]
    #     n = min(env.size(0), 8)

    #     all_imgs = [
    #         env[:n].narrow(start=0, length=self.imlength, dim=1)
    #             .contiguous().view(
    #             -1,
    #             self.input_channels,
    #             self.imsize,
    #             self.imsize
    #         ).transpose(2, 3)]

    #     for i in range(7):
    #         latent = self.model.sample_prior(self.batch_size, env)
    #         samples = self.model.decode(latent)
    #         all_imgs.extend([
    #             samples.view(
    #                 self.batch_size,
    #                 self.input_channels,
    #                 self.imsize,
    #                 self.imsize,
    #             )[:n].transpose(2, 3)])
    #     comparison = torch.cat(all_imgs)
    #     save_dir = osp.join(self.log_dir, 's%d.png' % epoch)
    #     save_image(comparison.data.cpu(), save_dir, nrow=8)

    def dump_reconstructions(self, epoch):
        self.dump_mixed_latents(epoch)
        batch, reconstructions = self.eval_data["test/last_batch"]
        obs = batch["x_t"]
        env = batch["env"]
        n = min(obs.size(0), 8)
        comparison = torch.cat([
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
    # def dump_reconstructions(self, epoch):
    #     self.dump_mixed_latents(epoch)
    #     batch, reconstructions, env_reconstructions = self.eval_data["test/last_batch"]
    #     obs = batch["x_t"]
    #     env = batch["env"]
    #     n = min(obs.size(0), 8)
    #     comparison = torch.cat([
    #         env[:n].narrow(start=0, length=self.imlength, dim=1)
    #             .contiguous().view(
    #             -1,
    #             3,
    #             self.imsize,
    #             self.imsize
    #         ).transpose(2, 3),
    #         obs[:n].narrow(start=0, length=self.imlength, dim=1)
    #             .contiguous().view(
    #             -1,
    #             3,
    #             self.imsize,
    #             self.imsize
    #         ).transpose(2, 3),
    #         reconstructions.view(
    #             self.batch_size,
    #             3,
    #             self.imsize,
    #             self.imsize,
    #         )[:n].transpose(2, 3),
    #         env_reconstructions.view(
    #             self.batch_size,
    #             3,
    #             self.imsize,
    #             self.imsize,
    #         )[:n].transpose(2, 3)
    #     ])
    #     save_dir = osp.join(self.log_dir, 'r%d.png' % epoch)
    #     save_image(comparison.data.cpu(), save_dir, nrow=n)

