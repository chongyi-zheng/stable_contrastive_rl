from __future__ import print_function
import copy
import torch.utils.data
from IPython.core.debugger import Pdb;

from rlkit.data_management.images import normalize_image
debug = Pdb().set_trace

# Adapted from pytorch examples

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.util.ml_util import ConstantSchedule
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger
import os.path as osp
from rlkit.envs.multitask.point2d import MultitaskImagePoint2DEnv
import numpy as np

class ACAI(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=84,
            added_fc_size=0,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            min_variance=1e-4,
            use_min_variance=True,
            state_size=0,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(128)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 512  # kernel 3

        # self.hidden = nn.Linear(self.conv_output_dim + added_fc_size, representation_size)

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)

        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3)
        self.conv5 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=3)
        self.conv6 = nn.ConvTranspose2d(32, input_channels, kernel_size=6, stride=3)

        # critic network
        self.conv7 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 64, kernel_size=5, stride=3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=5, stride=3)
        self.bn6 = nn.BatchNorm2d(128)
        # self.hidden = nn.Linear(self.conv_output_dim + added_fc_size, representation_size)
        self.fc4 = nn.Linear(self.conv_output_dim, representation_size)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)
        self.hidden_init(self.conv7.weight)
        self.conv7.bias.data.fill_(0)
        self.hidden_init(self.conv8.weight)
        self.conv8.bias.data.fill_(0)
        self.hidden_init(self.conv9.weight)
        self.conv9.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        self.hidden_init(self.fc4.weight)
        self.fc4.bias.data.fill_(0)
        self.fc4.weight.data.uniform_(-init_w, init_w)

    def encode(self, input):
        # debug()
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)

        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        h = x.view(-1, 512)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dimension=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = ptu.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 128, 2, 2)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1, self.imsize * self.imsize * self.input_channels)

        return self.sigmoid(x)

    def forward(self, x):
        batch_size = x.size()[0]
        x_2 = x[torch.randperm(batch_size).cuda()]
        mu, logvar = self.encode(x)
        mu_2, logvar_2 = self.encode(x_2)
        z = self.reparameterize(mu, logvar)
        z_2 = self.reparameterize(mu_2, logvar_2)
        alpha = torch.rand(batch_size, 1)
        alpha = 0.5 - torch.abs(0.5 - alpha)
        a = Variable(torch.from_numpy(np.diagflat(alpha.numpy()))).cuda()
        one_minus_a = Variable(torch.from_numpy(np.diagflat((1 - alpha).numpy()))).cuda()
        z_alpha = a.matmul(z) + one_minus_a.matmul(z_2)
        return self.decode(z), mu, logvar, self.critic(self.decode(z_alpha)).cuda(), \
               Variable(alpha).cuda()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        # TODO: is the deepcopy necessary?
        self.__dict__.update(copy.deepcopy(d))

    def critic(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)

        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        x = F.relu(self.bn4(self.conv7(x)))
        x = F.relu(self.bn5(self.conv8(x)))
        x = F.relu(self.bn6(self.conv9(x)))

        h = x.view(-1, 512)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dimension=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.sigmoid(self.fc4(h)) * 0.5
        return torch.mean(mu, 1)




class ACAITrainer():
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            imsize=84,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            state_sim_debug=False,
            mse_weight=0.1,
            is_auto_encoder=False,
            lmbda=0.5,
            mu=1,
            gamma=0.2,
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.imsize = imsize
        self.do_scatterplot = do_scatterplot
        self.lmbda = lmbda
        self.mu = mu
        self.gamma = gamma

        """
        I think it's a bit nicer if the caller makes this call, i.e.
        ```
        m = ConvVAE(representation_size)
        if ptu.gpu_enabled():
            m.cuda()
        t = ConvVAETrainer(train_data, test_data, m)
        ```
        However, I'll leave this here for backwards-compatibility.
        """
        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.normalize = normalize
        self.state_sim_debug = state_sim_debug
        self.mse_weight = mse_weight
        self.x_next_index = self.input_channels * self.imsize ** 2

        if self.normalize:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
        # self.train_dataset = ((self.train_dataset - self.train_data_mean)) + 1 / 2
        # self.test_dataset = ((self.test_dataset - self.train_data_mean)) + 1 / 2

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        samples = normalize_image(samples)
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        return ptu.np_to_var(samples)

    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.np_to_var(X), ptu.np_to_var(Y)

    def get_batch_smooth(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        samples = normalize_image(samples)
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        x_next, x = samples[:, :self.x_next_index], samples[:, self.x_next_index:]
        return ptu.np_to_var(x_next), ptu.np_to_var(x)

    def logprob(self, recon_x, x, mu, logvar):

        # Divide by batch_size rather than setting size_average=True because
        # otherwise the averaging will also happen across dimension 1 (the
        # pixels)
        return F.binary_cross_entropy(
            recon_x,
            x.narrow(start=0, length=self.imlength, dimension=1).contiguous().view(-1, self.imlength),
            size_average=False,
        ) / self.batch_size

    def kl_divergence(self, recon_x, x, mu, logvar):
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def state_similarity_loss(self, model, encoded_x, states):
        output = self.model.fc6(F.relu(self.model.fc5(encoded_x)))
        return torch.norm(output - states) ** 2 / self.batch_size

    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        self.model.train()
        losses = []
        bces = []
        kles = []
        mses = []
        losses_c = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(batches):
            data = self.get_batch()
            if sample_batch is not None:
                data = sample_batch(self.batch_size)

            self.optimizer.zero_grad()
            recon_batch, mu, logvar, predicted_alpha, alpha = self.model(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + beta * kle + self.lmbda * torch.norm(predicted_alpha, 2)
            regularizer_a = self.model.critic(self.gamma * data + (1 - self.gamma) * recon_batch)
            loss_c = torch.norm(predicted_alpha - alpha, 2) + self.mu * torch.norm(regularizer_a, 2)

            loss.backward(retain_graph=True)
            loss_c.backward()
            losses.append(loss.data[0])
            losses_c.append(loss_c.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])
            if self.state_sim_debug:
                mses.append(sim_loss.data[0])

            self.optimizer.step()
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.data[0] / len(data)))

        if not from_rl:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/critic_loss", np.mean(losses_c))
            logger.record_tabular("train/BCE", np.mean(bces))
            logger.record_tabular("train/KL", np.mean(kles))
            if self.state_sim_debug:
                logger.record_tabular("train/mse", np.mean(mses))
            logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_scatterplot=True,
            save_vae=True,
            from_rl=False,
    ):
        self.model.eval()
        losses = []
        losses_c = []
        bces = []
        kles = []
        zs = []
        mses = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(10):
            data = self.get_batch(train=False)
            recon_batch, mu, logvar, predicted_alpha, alpha = self.model(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + beta * kle + self.lmbda * torch.norm(predicted_alpha, 2)
            regularizer_a = self.model.critic(self.gamma * data + (1 - self.gamma) * recon_batch)
            loss_c = torch.norm(predicted_alpha - alpha, 2) + self.mu * torch.norm(regularizer_a, 2)

            z_data = ptu.get_numpy(mu.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.data[0])
            losses_c.append(loss_c.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])
            if self.state_sim_debug:
                mses.append(sim_loss.data[0])

            if batch_idx == 0 and save_reconstruction:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n].narrow(start=0, length=self.imlength, dimension=1)
                        .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ),
                    recon_batch.view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n]
                ])
                save_dir = osp.join(logger.get_snapshot_dir(), 'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)
        if self.do_scatterplot and save_scatterplot:
            self.plot_scattered(np.array(zs), epoch)

        if not from_rl:
            for key, value in self.debug_statistics().items():
                logger.record_tabular(key, value)

            logger.record_tabular("test/BCE", np.mean(bces))
            logger.record_tabular("test/KL", np.mean(kles))
            logger.record_tabular("test/loss", np.mean(losses))
            logger.record_tabular("test/critic_loss", np.mean(losses_c))
            logger.record_tabular("beta", beta)
            if self.state_sim_debug:
                logger.record_tabular("test/MSE", np.mean(mses))

            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)  # slow...

    # logdir = logger.get_snapshot_dir()
    # filename = osp.join(logdir, 'params.pkl')
    # torch.save(self.model, filename)

    def debug_statistics(self):
        """
        Given an image $$x$$, samples a bunch of latents from the prior
        $$z_i$$ and decode them $$\hat x_i$$.
        Compare this to $$\hat x$$, the reconstruction of $$x$$.
        Ideally
         - All the $$\hat x_i$$s do worse than $$\hat x$$ (makes sure VAE
           isn’t ignoring the latent)
         - Some $$\hat x_i$$ do better than other $$\hat x_i$$ (tests for
           coverage)
        """
        debug_batch_size = 64

        data = self.get_batch(train=False)
        recon_batch, mu, logvar, predicted_alpha, alpha = self.model(data)
        img = data[0]
        recon_mse = ((recon_batch[0] - img) ** 2).mean()

        img_repeated = img.expand((debug_batch_size, img.shape[0]))

        samples = ptu.Variable(torch.randn(debug_batch_size, self.representation_size))
        random_imgs = self.model.decode(samples)
        random_mse = ((random_imgs - img_repeated) ** 2).mean(dim=1)

        mse_improvement = ptu.get_numpy(random_mse - recon_mse)
        stats = create_stats_ordered_dict(
            'debug/MSE improvement over random',
            mse_improvement,
        )
        stats['debug/MSE of random reconstruction'] = ptu.get_numpy(
            recon_mse
        )[0]
        return stats

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.Variable(torch.randn(64, self.representation_size))
        sample = self.model.decode(sample).cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize),
            save_dir
        )

    def plot_scattered(self, z, epoch):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.log(__file__ + ": Unable to load matplotlib. Consider "
                                  "setting do_scatterplot to False")
            return
        dim_and_stds = [(i, np.std(z[:, i])) for i in range(z.shape[1])]
        dim_and_stds = sorted(
            dim_and_stds,
            key=lambda x: x[1]
        )
        dim1 = dim_and_stds[-1][0]
        dim2 = dim_and_stds[-2][0]
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, dim1], z[:, dim2], marker='o', edgecolor='none')
        if self.model.dist_mu is not None:
            x1 = self.model.dist_mu[dim1:dim1 + 1]
            y1 = self.model.dist_mu[dim2:dim2 + 1]
            x2 = self.model.dist_mu[dim1:dim1 + 1] + self.model.dist_std[dim1:dim1 + 1]
            y2 = self.model.dist_mu[dim2:dim2 + 1] + self.model.dist_std[dim2:dim2 + 1]
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        axes.set_title('dim {} vs dim {}'.format(dim1, dim2))
        plt.grid(True)
        save_file = osp.join(logger.get_snapshot_dir(), 'scatter%d.png' % epoch)
        plt.savefig(save_file)