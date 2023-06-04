import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN
from rlkit.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE
from rlkit.torch.vae.conv_vae import ConvVAE

class ConditionalConvVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):
        """
        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = 6
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture
        self.reconstruction_channels = reconstruction_channels
        self.decoder_output_activation = decoder_output_activation

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        # self.encoder=encoder_class(
        #     **conv_args,
        #     paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        #     input_height=self.imsize,
        #     input_width=self.imsize,
        #     input_channels=self.input_channels,
        #     output_size=conv_output_size,
        #     init_w=init_w,
        #     hidden_init=hidden_init,
        #     **conv_kwargs)

        # self.decoder = decoder_class(
        #     **deconv_args,
        #     fc_input_size=representation_size,
        #     init_w=init_w,
        #     output_activation=decoder_output_activation,
        #     paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
        #     hidden_init=hidden_init,
        #     **deconv_kwargs)

        self.relu = nn.LeakyReLU()
        self.gain = weight_init_gain
        self.init_w = init_w

        self.base_depth = base_depth
        self.epoch = 0
        self.decoder_distribution=decoder_distribution
        self.representation_size = representation_size

        self._create_layers()

    def _create_layers(self):
        self.conv1a = nn.Conv2d(3, self.base_depth, 3, stride=3)
        nn.init.xavier_uniform_(self.conv1a.weight, gain=self.gain)
        self.conv2a = nn.Conv2d(self.base_depth, self.base_depth * 2 , 3, stride=3)
        nn.init.xavier_uniform_(self.conv2a.weight, gain=self.gain)

        self.conv1b = nn.Conv2d(3, self.base_depth, 3, stride=3)
        nn.init.xavier_uniform_(self.conv1b.weight, gain=self.gain)
        self.conv2b = nn.Conv2d(self.base_depth, self.base_depth * 2 , 3, stride=3)
        nn.init.xavier_uniform_(self.conv2b.weight, gain=self.gain)

        self.conv3 = nn.Conv2d(2 * self.base_depth* 2, self.base_depth * 4, 3, stride=2) # fusion
        nn.init.xavier_uniform_(self.conv3.weight, gain=self.gain)

        self.fc1 = nn.Linear(self.base_depth*4*2*2, self.representation_size)
        self.fc2 = nn.Linear(self.base_depth*4*2*2, self.representation_size)

        self.fc1.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc1.bias.data.uniform_(-self.init_w, self.init_w)

        self.fc2.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc2.bias.data.uniform_(-self.init_w, self.init_w)

        self.deconv_fc1 = nn.Linear(self.representation_size, 2*2*self.base_depth*4)
        self.deconv_fc1.weight.data.uniform_(-self.init_w, self.init_w)
        self.deconv_fc1.bias.data.uniform_(-self.init_w, self.init_w)

        self.dconv1 = nn.ConvTranspose2d(self.base_depth*4, self.base_depth*4, 5, stride=3)
        nn.init.xavier_uniform_(self.dconv1.weight, gain=self.gain)
        self.dconv2 = nn.ConvTranspose2d(2*self.base_depth*4, self.base_depth*2, 6, stride=2) # skip connection
        nn.init.xavier_uniform_(self.dconv2.weight, gain=self.gain)

        self.dconv3 = nn.ConvTranspose2d(2*self.base_depth*2, 3, 10, stride=2)
        nn.init.xavier_uniform_(self.dconv3.weight, gain=self.gain)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=4)


    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        input = input.view(-1, self.input_channels, self.imsize, self.imsize)
        # import pdb; pdb.set_trace()
        x = input[:, :3, :, :]
        x0 = input[:, 3:, :, :]

        a1 = self.conv1a(x)
        a2 = self.conv2a(self.relu(a1))
        b1 = self.conv1b(x0)
        b2 = self.conv2b(self.relu(b1)) # 32 x 18 x 18

        h2 = torch.cat((a2, b2), dim=1)
        h3 = self.conv3(self.relu(h2))

        # hlayers = [a1, h2, h3, ]
        # for l in hlayers:
        #     print(l.shape)
        h = h3.view(h3.size()[0], -1)

        ### encode
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        latent_distribution_params = (mu, logvar)

        ### reparameterize

        latents = self.reparameterize(latent_distribution_params)

        dh0 = self.deconv_fc1(latents)
        dh0 = self.relu(dh0.view(-1, self.base_depth*4, 2, 2))
        ### decode

        dh1 = self.dconv1(dh0)
        dh1 = torch.cat((dh1, self.up2(dh0)), dim=1)

        dh2 = self.dconv2(self.relu(dh1))

        # fusion
        f = torch.cat((dh2, self.up1(b2)), dim=1)

        dh3 = self.dconv3(self.relu(f))

        # print(dh3.shape)

        """
        f3 = torch.cat((dh3, b2), dim=1)
        dh4 = self.dconv4(self.relu(f3))
        dh5 = self.dconv5(self.relu(dh4))
        """

        # dlayers = [dh1, dh2, dh3, dh4, dh5]
        # for l in dlayers:
        #     print(l.shape)

        decoded = self.decoder_output_activation(dh3)

        decoded = decoded.view(-1, self.imsize*self.imsize*self.reconstruction_channels)
        # if self.decoder_distribution == 'bernoulli': # assume bernoulli
        reconstructions, obs_distribution_params = decoded, [decoded]

        return reconstructions, obs_distribution_params, latent_distribution_params

    def encode(self, input):
        input = input.view(-1, self.input_channels, self.imsize, self.imsize)
        x = input[:, :3, :, :]
        x0 = input[:, 3:, :, :]
        a1 = self.conv1a(x)
        a2 = self.conv2a(self.relu(a1))
        b1 = self.conv1b(x0)
        b2 = self.conv2b(self.relu(b1)) # 32 x 18 x 18



        h2 = torch.cat((a2, b2), dim=1)
        h3 = self.conv3(self.relu(h2))

        # hlayers = [h2, h3, h4, h5]
        # for l in hlayers:
        #     print(l.shape)
        h = h3.view(h3.size()[0], -1)

        ### encode
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return (mu, logvar)


    def decode(self, latents, data):
        data = data.view(-1, self.input_channels, self.imsize, self.imsize)
        x0 = data[:, 3:, :, :]

        b1 = self.conv1b(x0)
        b2 = self.conv2b(self.relu(b1)) # 32 x 18 x 18

        # newer arch
        dh0 = self.deconv_fc1(latents)
        dh0 = self.relu(dh0.view(-1, self.base_depth*4, 2, 2))
        ### decode

        dh1 = self.dconv1(dh0)
        dh1 = torch.cat((dh1, self.up2(dh0)), dim=1)

        dh2 = self.dconv2(self.relu(dh1))

        # fusion
        if dh2.shape != self.up1(b2).shape:
            import pdb; pdb.set_trace()
        f = torch.cat((dh2, self.up1(b2)), dim=1)

        dh3 = self.dconv3(self.relu(f))


        # older arch
        # dh0 = self.deconv_fc1(latents)
        # dh0 = self.relu(dh0.view(-1, self.base_depth*4, 2, 2))
        # dh1 = self.dconv1(dh0)
        # dh1 = torch.cat((dh1, self.up2(dh0)), dim=1)
        # dh2 = self.dconv2(self.relu(dh1))

        # f = torch.cat((dh2, self.up1(b2)), dim=1)

        # dh3 = self.dconv3(self.relu(f))



        #f3 = torch.cat((dh3, b2), dim=1)
        #dh4 = self.dconv4(self.relu(f3))
        #dh5 = self.dconv5(self.relu(dh4))

        #decoded = self.decoder_output_activation(dh5)
        decoded = self.decoder_output_activation(dh3)
        decoded = decoded.view(-1, self.imsize*self.imsize*self.reconstruction_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.view(
                -1, self.input_channels, self.imsize, self.imsize
            )
            length = self.reconstruction_channels * self.imsize * self.imsize
            x = inputs[:, :self.reconstruction_channels, :, :].view(-1, length)
            # x = x.narrow(start=0, length=length,
                 # dim=1).contiguous().view(-1, length)
            reconstruction_x = obs_distribution_params[0]
            log_prob = compute_bernoulli_log_prob(x, reconstruction_x) * (self.imsize*self.imsize*self.reconstruction_channels)
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength // 2,
                                   dim=1).contiguous().view(-1, self.imlength // 2)
            log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0],reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))


class CVAE0(GaussianLatentVAE):
    def __init__(
            self,
            latent_sizes,
            architecture,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            num_labels = 0,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            add_labels_to_latents=False,
            hidden_init=nn.init.xavier_uniform_,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):
        """
        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        representation_size = latent_sizes[0] + latent_sizes[1]
        super().__init__(representation_size)
        self.latent_sizes = latent_sizes
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.gain = weight_init_gain
        self.relu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.5)
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture
        self.reconstruction_channels = reconstruction_channels
        self.decoder_output_activation = decoder_output_activation

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        self.cond_encoder=encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.encoder=encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=self.representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.c = nn.Linear(self.encoder.output_size, latent_sizes[1])
        self.z = nn.Linear(self.encoder.output_size, self.representation_size)
        self.z_mu = nn.Linear(self.representation_size + latent_sizes[1], latent_sizes[0])
        self.z_var = nn.Linear(self.representation_size + latent_sizes[1], latent_sizes[0])
        self.bn_z = nn.BatchNorm1d(self.representation_size)
        self.bn_c = nn.BatchNorm1d(latent_sizes[1])

        nn.init.xavier_uniform_(self.z.weight, gain=self.gain)
        nn.init.xavier_uniform_(self.c.weight, gain=self.gain)
        nn.init.xavier_uniform_(self.z_mu.weight, gain=self.gain)
        nn.init.xavier_uniform_(self.z_var.weight, gain=self.gain)

        self.z.bias.data.uniform_(-init_w, init_w)
        self.c.bias.data.uniform_(-init_w, init_w)
        self.z_mu.bias.data.uniform_(-init_w, init_w)
        self.z_var.bias.data.uniform_(-init_w, init_w)
        self.prior_mu, self.prior_logvar = None, None

        self.epoch = 0
        self.decoder_distribution=decoder_distribution

    def forward(self, x_t, x_0):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(x_t, x_0)
        latents = self.reparameterize(latent_distribution_params)
        obs_recononstruction, obs_distribution_params = self.decode(latents)
        return obs_recononstruction, obs_distribution_params, latent_distribution_params

    def encode(self, x_t, x_0, distrib=True):
        if x_0.shape[0] == 1:
            batch_size = len(x_t)
            x_0 = x_0.repeat(batch_size, 1)

        latents = self.dropout(self.relu(self.bn_z(self.z(self.dropout(self.encoder(x_t))))))

        conditioning = self.bn_c(self.c(self.dropout(self.cond_encoder(x_0))))
        cond_latents = torch.cat([latents, conditioning], dim=1)
        mu = self.z_mu(cond_latents)

        if not distrib: return torch.cat([mu, conditioning], dim=1)

        if self.log_min_variance is None:
            logvar = self.z_var(cond_latents)
        else:
            logvar = self.log_min_variance + torch.abs(self.z_var(cond_latents))

        return (mu, logvar, conditioning)

    def reparameterize(self, latent_distribution_params):
        if self.training:
            mu = self.rsample((latent_distribution_params[0], latent_distribution_params[1]))
        else:
            mu = latent_distribution_params[0]
        return torch.cat([mu, latent_distribution_params[2]], dim=1)

    def update_prior(self, mu, logvar):
        self.prior_mu = mu
        self.prior_logvar = logvar

    def sample_prior(self, batch_size, x_0, true_prior=True):
        if x_0.shape[0] == 1:
            x_0 = x_0.repeat(batch_size, 1)

        z_sample = ptu.randn(batch_size, self.latent_sizes[0])

        if not true_prior:
            stds = np.exp(0.5 * self.prior_logvar)
            z_sample = z_sample * stds + self.prior_mu

        conditioning = self.bn_c(self.c(self.dropout(self.cond_encoder(x_0))))
        cond_sample = torch.cat([z_sample, conditioning], dim=1)
        return cond_sample


    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params[0], latent_distribution_params[1]
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        return torch.cat([latent_distribution_params[0],latent_distribution_params[2]], dim=1).cpu()


    def decode(self, latents):
        decoded = self.decoder(latents).view(-1, self.imlength)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params, mean=True):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                dim=1).contiguous().view(-1, self.imlength)
            if mean:
                log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0]) * self.imlength
            else:
                log_prob = -1 * F.binary_cross_entropy(inputs, obs_distribution_params[0]) * self.imlength
                return 1/0 #NOT SURE ABOVE IS ROW WISE MEAN, MAKE SURE THIS IS CORRECT AND DOES SAME AS BELOW CASE
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength, dim=1).contiguous().view(-1, self.imlength)
            if mean:
                log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='elementwise_mean')
            else:
                log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='none').mean(dim=1, keepdim=True)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

class CVAE(GaussianLatentVAE):
    def __init__(
            self,
            latent_sizes,
            architecture,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            num_labels = 0,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            add_labels_to_latents=False,
            hidden_init=nn.init.xavier_uniform_,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):
        """
        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param hidden_init:
        """
        representation_size = latent_sizes[0] + latent_sizes[1]
        super().__init__(representation_size)
        self.latent_sizes = latent_sizes
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.gain = weight_init_gain
        self.relu = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.5)
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture
        self.reconstruction_channels = reconstruction_channels
        self.decoder_output_activation = decoder_output_activation

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        self.cond_encoder=encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.encoder=encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels*2,
            output_size=conv_output_size,
            init_w=init_w,
            hidden_init=hidden_init,
            **conv_kwargs)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=self.representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.c = nn.Linear(self.encoder.output_size, latent_sizes[1])        
        self.z_mu = nn.Linear(self.encoder.output_size, latent_sizes[0])
        self.z_var = nn.Linear(self.encoder.output_size, latent_sizes[0])
        self.bn_c = nn.BatchNorm1d(latent_sizes[1])

        nn.init.xavier_uniform_(self.c.weight, gain=self.gain)
        nn.init.xavier_uniform_(self.z_mu.weight, gain=self.gain)
        nn.init.xavier_uniform_(self.z_var.weight, gain=self.gain)

        self.c.bias.data.uniform_(-init_w, init_w)
        self.z_mu.bias.data.uniform_(-init_w, init_w)
        self.z_var.bias.data.uniform_(-init_w, init_w)
        self.prior_mu, self.prior_logvar = None, None

        self.epoch = 0
        self.decoder_distribution=decoder_distribution

    def forward(self, x_t, x_0):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.encode(x_t, x_0)
        latents = self.reparameterize(latent_distribution_params)
        obs_recononstruction, obs_distribution_params = self.decode(latents)
        return obs_recononstruction, obs_distribution_params, latent_distribution_params

    def encode(self, x_t, x_0, distrib=True):
        batch_size = len(x_t)
        if x_0.shape[0] == 1:
            x_0 = x_0.repeat(batch_size, 1)

        x_pos = x_t.reshape(-1, self.input_channels, self.imsize, self.imsize)
        x_obj = x_0.reshape(-1, self.input_channels, self.imsize, self.imsize)
        
        comb_obs = torch.cat([x_pos, x_obj], dim=1).reshape(batch_size, -1)

        z = self.dropout(self.encoder(comb_obs))
        mu = self.z_mu(z)

        conditioning = self.bn_c(self.c(self.dropout(self.cond_encoder(x_0))))

        if not distrib: return torch.cat([mu, conditioning], dim=1)

        if self.log_min_variance is None:
            logvar = self.z_var(z)
        else:
            logvar = self.log_min_variance + torch.abs(self.z_var(z))

        return (mu, logvar, conditioning)

    def reparameterize(self, latent_distribution_params):
        if self.training:
            mu = self.rsample((latent_distribution_params[0], latent_distribution_params[1]))
        else:
            mu = latent_distribution_params[0]
        return torch.cat([mu, latent_distribution_params[2]], dim=1)

    def update_prior(self, mu, logvar):
        self.prior_mu = mu
        self.prior_logvar = logvar

    def sample_prior(self, batch_size, x_0, true_prior=True):
        if x_0.shape[0] == 1:
            x_0 = x_0.repeat(batch_size, 1)

        z_sample = ptu.randn(batch_size, self.latent_sizes[0])

        if not true_prior:
            stds = np.exp(0.5 * self.prior_logvar)
            z_sample = z_sample * stds + self.prior_mu

        conditioning = self.bn_c(self.c(self.dropout(self.cond_encoder(x_0))))
        cond_sample = torch.cat([z_sample, conditioning], dim=1)
        return cond_sample


    def kl_divergence(self, latent_distribution_params):
        mu, logvar = latent_distribution_params[0], latent_distribution_params[1]
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def get_encoding_from_latent_distribution_params(self, latent_distribution_params):
        return torch.cat([latent_distribution_params[0],latent_distribution_params[2]], dim=1).cpu()


    def decode(self, latents):
        decoded = self.decoder(latents).view(-1, self.imlength)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params, mean=True):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                dim=1).contiguous().view(-1, self.imlength)
            if mean:
                log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0]) * self.imlength
            else:
                log_prob = -1 * F.binary_cross_entropy(inputs, obs_distribution_params[0]) * self.imlength
                return 1/0 #NOT SURE ABOVE IS ROW WISE MEAN, MAKE SURE THIS IS CORRECT AND DOES SAME AS BELOW CASE
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength, dim=1).contiguous().view(-1, self.imlength)
            if mean:
                log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='elementwise_mean')
            else:
                log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='none').mean(dim=1, keepdim=True)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))


class DeltaCVAE(CVAE):
    def __init__(
            self,
            latent_sizes,
            architecture,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            num_labels=0,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            add_labels_to_latents=False,
            hidden_init=nn.init.xavier_uniform_,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):
        super().__init__(
            latent_sizes,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain, )

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        self.cond_decoder = decoder_class(
            **deconv_args,
            fc_input_size=latent_sizes[1],
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

    def forward(self, x_t, x_0):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        z_distrib = self.encode(x_t, x_0)
        z = self.reparameterize(z_distrib)
        x_recon, x_distrib = self.decode(z)
        env_recon, env_distrib = self.decode(z_distrib, conditioning=True)
        return (x_recon, x_distrib, z_distrib), (env_recon, env_distrib)

    def decode(self, latents, conditioning=False):
        if conditioning:
            decoded = self.cond_decoder(latents[2]).view(-1, self.imlength)
        else:
            decoded = self.decoder(latents).view(-1, self.imlength)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))


class ACE(CVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            num_labels = 0,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            add_labels_to_latents=False,
            hidden_init=nn.init.xavier_uniform_,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
        ):
        super().__init__(
            representation_size,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain)

        self.CVAE = CVAE(
            representation_size,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain)

        self.adversary = ConvVAE(
            representation_size[0],
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            input_channels,
            imsize,
            init_w,
            min_variance,
            hidden_init)

    def forward(self, x_t, x_0):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.CVAE.encode(x_t, x_0)
        latents = self.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.CVAE.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params

    def encode(self, x_t, x_0, distrib=True):
        return self.CVAE.encode(x_t, x_0, distrib)

    def decode(self, latents):
        return self.CVAE.decode(latents)

    def logprob(self, inputs, obs_distribution_params):
        return self.CVAE.logprob(inputs, obs_distribution_params)

    def sample_prior(self, batch_size, x_0):
        return self.CVAE.sample_prior(batch_size, x_0)



class CDVAE(CVAE):#CHANGE SO ONLY SEES Z -> DYNAMICS INDPENEDENT OF X_0
    "Conditional Dynamics VAE"
    def __init__(
            self,
            representation_size,
            architecture,
            action_dim,
            dynamics_type=None,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            input_channels=1,
            num_labels=0,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            add_labels_to_latents=False,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):

        super().__init__(
            representation_size,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain
        )

        self.action_dim = action_dim
        self.dynamics_type = dynamics_type

        self.globally_linear = nn.Linear(self.latent_sizes + self.action_dim, self.latent_sizes)

        self.locally_linear_f1 = nn.Linear(self.latent_sizes, 400)
        self.locally_linear_f2 = nn.Linear(400, (self.latent_sizes + self.action_dim) * self.latent_sizes)

        self.nonlinear_dynamics_f1 = nn.Linear(self.latent_sizes + self.action_dim, 400)
        self.nonlinear_dynamics_f2 = nn.Linear(400, self.latent_sizes)

        self.bn_d = nn.BatchNorm1d(400)
        self.do_d = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.globally_linear.weight, gain=self.gain)

        nn.init.xavier_uniform_(self.locally_linear_f1.weight, gain=self.gain)
        self.locally_linear_f1.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.locally_linear_f2.weight, gain=self.gain)
        self.locally_linear_f2.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.nonlinear_dynamics_f1.weight, gain=self.gain)
        self.nonlinear_dynamics_f1.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.nonlinear_dynamics_f2.weight, gain=self.gain)
        self.nonlinear_dynamics_f2.bias.data.uniform_(-init_w, init_w)


    def process_dynamics(self, latents, actions):
        if self.dynamics_type == 'global':
            return self.global_linear_dynamics(latents, actions)
        if self.dynamics_type == 'local':
            return self.local_linear_dynamics(latents, actions)
        if self.dynamics_type == 'nonlinear':
            return self.nonlinear_dynamics(latents, actions)

    def global_linear_dynamics(self, latents, actions):
        action_obs_pair = torch.cat([latents, actions], dim=1)
        z_prime = self.globally_linear(action_obs_pair)
        return z_prime

    def local_linear_dynamics(self, latents, actions):
        output = self.locally_linear_f2(self.relu(self.locally_linear_f1(latents)))
        dynamics = output.view(latents.shape[0], self.latent_sizes, self.latent_sizes + self.action_dim)

        z_prime = ptu.zeros_like(latents)
        action_obs_pair = torch.cat([latents, actions], dim=1)
        for i in range(latents.shape[0]):
            z_prime[i] = torch.matmul(dynamics[i], action_obs_pair[i])
        return z_prime

    def nonlinear_dynamics(self, latents, actions):
        action_obs_pair = torch.cat([latents, actions], dim=1)
        z_prime = self.nonlinear_dynamics_f2(self.bn_d(self.relu(self.nonlinear_dynamics_f1(action_obs_pair))))
        return z_prime

class DeltaDynamicsCVAE(DeltaCVAE):#CHANGE SO ONLY SEES Z -> DYNAMICS INDPENEDENT OF X_0
    "Conditional Dynamics VAE"
    def __init__(
            self,
            representation_size,
            architecture,
            action_dim,
            dynamics_type=None,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            input_channels=1,
            num_labels=0,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            add_labels_to_latents=False,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):

        super().__init__(
            representation_size,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain
        )

        self.action_dim = action_dim
        self.dynamics_type = dynamics_type

        self.globally_linear = nn.Linear(self.latent_sizes + self.action_dim, self.latent_sizes)

        self.locally_linear_f1 = nn.Linear(self.latent_sizes, 400)
        self.locally_linear_f2 = nn.Linear(400, (self.latent_sizes + self.action_dim) * self.latent_sizes)

        self.nonlinear_dynamics_f1 = nn.Linear(self.latent_sizes + self.action_dim, 400)
        self.nonlinear_dynamics_f2 = nn.Linear(400, self.latent_sizes)

        self.bn_d = nn.BatchNorm1d(400)
        self.do_d = nn.Dropout(0.2)

        nn.init.xavier_uniform_(self.globally_linear.weight, gain=self.gain)

        nn.init.xavier_uniform_(self.locally_linear_f1.weight, gain=self.gain)
        self.locally_linear_f1.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.locally_linear_f2.weight, gain=self.gain)
        self.locally_linear_f2.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.nonlinear_dynamics_f1.weight, gain=self.gain)
        self.nonlinear_dynamics_f1.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.nonlinear_dynamics_f2.weight, gain=self.gain)
        self.nonlinear_dynamics_f2.bias.data.uniform_(-init_w, init_w)


    def process_dynamics(self, latents, actions):
        if self.dynamics_type == 'global':
            return self.global_linear_dynamics(latents, actions)
        if self.dynamics_type == 'local':
            return self.local_linear_dynamics(latents, actions)
        if self.dynamics_type == 'nonlinear':
            return self.nonlinear_dynamics(latents, actions)

    def global_linear_dynamics(self, latents, actions):
        action_obs_pair = torch.cat([latents, actions], dim=1)
        z_prime = self.globally_linear(action_obs_pair)
        return z_prime

    def local_linear_dynamics(self, latents, actions):
        output = self.locally_linear_f2(self.relu(self.locally_linear_f1(latents)))
        dynamics = output.view(latents.shape[0], self.latent_sizes, self.latent_sizes + self.action_dim)

        z_prime = ptu.zeros_like(latents)
        action_obs_pair = torch.cat([latents, actions], dim=1)
        for i in range(latents.shape[0]):
            z_prime[i] = torch.matmul(dynamics[i], action_obs_pair[i])
        return z_prime

    def nonlinear_dynamics(self, latents, actions):
        action_obs_pair = torch.cat([latents, actions], dim=1)
        z_prime = self.nonlinear_dynamics_f2(self.bn_d(self.relu(self.nonlinear_dynamics_f1(action_obs_pair))))
        return z_prime

class CADVAE(CVAE):
    "Conditioned Adversarial Dynamics VAE"

    def __init__(
            self,
            representation_size,
            architecture,
            action_dim,
            dynamics_type=None,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            input_channels=1,
            num_labels=0,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=nn.init.xavier_uniform_,
            add_labels_to_latents=False,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,
    ):
        super().__init__(
            representation_size,
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            num_labels,
            input_channels,
            imsize,
            init_w,
            min_variance,
            add_labels_to_latents,
            hidden_init,
            reconstruction_channels,
            base_depth,
            weight_init_gain)

        self.CDVAE = CDVAE(
            representation_size,
            architecture,
            action_dim,
            dynamics_type,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            input_channels,
            num_labels,
            imsize,
            init_w,
            min_variance,
            hidden_init,
            add_labels_to_latents,
            reconstruction_channels,
            base_depth,
            weight_init_gain)

        self.adversary = ConvVAE(
            representation_size[0],
            architecture,
            encoder_class,
            decoder_class,
            decoder_output_activation,
            decoder_distribution,
            input_channels,
            imsize,
            init_w,
            min_variance,
            hidden_init)

    def forward(self, x_t, x_0):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        latent_distribution_params = self.CDVAE.encode(x_t, x_0)
        latents = self.CDVAE.reparameterize(latent_distribution_params)
        reconstructions, obs_distribution_params = self.CDVAE.decode(latents)
        return reconstructions, obs_distribution_params, latent_distribution_params

    def encode(self, x_t, x_0, distrib=True):
        return self.CDVAE.encode(x_t, x_0, distrib)

    def decode(self, latents, conditioning=False):
        return self.CDVAE.decode(latents, conditioning)

    def logprob(self, inputs, obs_distribution_params):
        return self.CDVAE.logprob(inputs, obs_distribution_params)

    def sample_prior(self, batch_size, x_0):
        return self.CDVAE.sample_prior(batch_size, x_0)

    def process_dynamics(self, latents, actions):
        return self.CDVAE.process_dynamics(latents, actions)
