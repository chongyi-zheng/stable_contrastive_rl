import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from rlkit.pythonplusplus import identity
from rlkit.torch import pytorch_util as ptu
import numpy as np
from rlkit.torch.networks import CNN, TwoHeadDCNN, DCNN
from rlkit.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE

###### DEFAULT ARCHITECTURES #########

imsize48_default_architecture=dict(
        conv_args = dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[],
            #conv_normalization_type="batch",
            #fc_normalization_type="batch",
        ),
        deconv_args=dict(
            hidden_sizes=[],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3,3],
            n_channels=[32, 16],
            strides=[2,2],
        ),
        deconv_kwargs=dict(
            deconv_normalization_type="batch",
            fc_normalization_type="batch",
        )
    )


imsize48_default_architecture_spatial=dict(
        conv_args = dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 8, 4],
            strides=[3, 1, 1],
        ),
        conv_kwargs=dict(
            hidden_sizes=[],
            conv_normalization_type="batch",
            fc_normalization_type="batch",
            output_conv_channels=True,
        ),
        deconv_args=dict(
            hidden_sizes=[],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3,3],
            n_channels=[32, 16],
            strides=[2,2],
        ),
        deconv_kwargs=dict(
            deconv_normalization_type="batch",
            fc_normalization_type="batch",
        )
    )


imsize48_default_architecture_with_more_hidden_layers = dict(
        conv_args=dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[500, 300, 150],
            conv_normalization_type="batch",
            fc_normalization_type="batch",
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3, 3],
            n_channels=[32, 16],
            strides=[2, 2],
        ),
        deconv_kwargs=dict(
            deconv_normalization_type="batch",
            fc_normalization_type="batch",
        )
    )

imsize84_default_architecture=dict(
        conv_args = dict(
            kernel_sizes=[5, 5, 5],
            n_channels=[16, 32, 32],
            strides=[3, 3, 3],
        ),
        conv_kwargs=dict(
            hidden_sizes=[],
            conv_normalization_type="batch",
            fc_normalization_type="batch",
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=2,
            deconv_input_height=2,
            deconv_input_channels=32,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[5,6],
            n_channels=[32, 16],
            strides=[3,3],
        ),
        deconv_kwargs=dict(
            deconv_normalization_type="batch",
            fc_normalization_type="batch",
        )
    )

imsize84_default_architecture_with_more_hidden_layers=dict(
        conv_args = dict(
            kernel_sizes=[5, 5, 5],
            n_channels=[16, 32, 32],
            strides=[3, 3, 3],
        ),
        conv_kwargs=dict(
            hidden_sizes=[500, 300, 150],
            conv_normalization_type="batch",
            fc_normalization_type="batch",
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=2,
            deconv_input_height=2,
            deconv_input_channels=32,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[5,6],
            n_channels=[32, 16],
            strides=[3,3],
        ),
        deconv_kwargs=dict(
            deconv_normalization_type="batch",
            fc_normalization_type="batch",
        )
    )


class ConvVAE(GaussianLatentVAE):
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
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

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

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        self.fc1.bias.data.uniform_(-init_w, init_w)

        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.epoch = 0
        self.decoder_distribution=decoder_distribution

    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)

    def decode(self, latents):
        decoded = self.decoder(latents).view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                 dim=1).contiguous().view(-1, self.imlength)
            log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0]) * self.imlength
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0], reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))


class ConvDynamicsVAE(ConvVAE):
    def __init__(
            self,
            representation_size,
            architecture,
            action_dim,
            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',
            input_channels=1,
            imsize=48,
            dynamics_type=None,
            init_w=1e-3,
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
    ):

        super().__init__(
            representation_size,
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

        self.action_dim = action_dim
        self.dynamics_type = dynamics_type

        self.globally_linear = nn.Linear(representation_size + self.action_dim, representation_size)

        self.locally_linear_f1 = nn.Linear(representation_size, 400)
        self.locally_linear_f2 = nn.Linear(400, (representation_size + self.action_dim) * representation_size)

        self.nonlinear_dynamics_f1 = nn.Linear(representation_size + self.action_dim, 400)
        self.nonlinear_dynamics_f2 = nn.Linear(400, representation_size)

        self.globally_linear.weight.data.uniform_(-init_w, init_w)

        self.locally_linear_f1.weight.data.uniform_(-init_w, init_w)
        self.locally_linear_f1.bias.data.uniform_(-init_w, init_w)

        self.locally_linear_f2.weight.data.uniform_(-init_w, init_w)
        self.locally_linear_f2.bias.data.uniform_(-init_w, init_w)

        self.nonlinear_dynamics_f1.weight.data.uniform_(-init_w, init_w)
        self.nonlinear_dynamics_f1.bias.data.uniform_(-init_w, init_w)

        self.nonlinear_dynamics_f2.weight.data.uniform_(-init_w, init_w)
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
        output = self.locally_linear_f2(F.relu(self.locally_linear_f1(latents)))
        dynamics = output.view(latents.shape[0], self.representation_size, self.representation_size + self.action_dim)

        z_prime = ptu.zeros_like(latents)
        action_obs_pair = torch.cat([latents, actions], dim=1)
        for i in range(latents.shape[0]):
            z_prime[i] = torch.matmul(dynamics[i], action_obs_pair[i])
        return z_prime

    def nonlinear_dynamics(self, latents, actions):
        action_obs_pair = torch.cat([latents, actions], dim=1)
        z_prime = self.nonlinear_dynamics_f2(F.relu(self.nonlinear_dynamics_f1(action_obs_pair)))
        return z_prime


class ConvVAEDouble(ConvVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=TwoHeadDCNN,
            decoder_output_activation=identity,
            decoder_distribution='gaussian',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-4,
            hidden_init=ptu.fanin_init,
            min_log_clamp=0,
    ):
        super().__init__(
            representation_size,
            architecture,

            encoder_class=encoder_class,
            decoder_class=decoder_class,
            decoder_output_activation=decoder_output_activation,
            decoder_distribution=decoder_distribution,

            input_channels=input_channels,
            imsize=imsize,
            init_w=init_w,
            min_variance=min_variance,
            hidden_init=hidden_init,
        )
        self.min_log_var = min_log_clamp

    def decode(self, latents):
        first_output, second_output = self.decoder(latents)
        first_output = first_output.view(-1, self.imsize*self.imsize*self.input_channels)
        second_output = second_output.view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'gaussian':
            return first_output, (first_output, second_output)
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'gaussian':
            dec_mu, dec_logvar = obs_distribution_params
            dec_mu = dec_mu.view(-1, self.imlength)
            dec_var = dec_logvar.view(-1, self.imlength).exp()
            inputs = inputs.view(-1, self.imlength)
            log_prob = compute_gaussian_log_prob(inputs, dec_mu, dec_var)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

class AutoEncoder(ConvVAE):
    def forward(self, x):
        mu, logvar = self.encode(input)
        reconstructions, obs_distribution_params = self.decode(mu)
        return reconstructions, obs_distribution_params, (mu, logvar)


class SpatialAutoEncoder(ConvVAE):
    def __init__(
            self,
            representation_size,
            *args,
            temperature=1.0,
            use_softmax=True,
            encode_feature_points=True,
            **kwargs
    ):
        num_feat_points = representation_size // 2
        # Override the number of channels in the architecture to match the
        # number of feature points
        kwargs['architecture']["conv_args"]["n_channels"][-1] = num_feat_points

        super().__init__(representation_size, *args, **kwargs)
        assert self.architecture["conv_args"]["n_channels"][-1] == num_feat_points

        self.fc1 = nn.Linear(representation_size, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        self.fc1.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc1.bias.data.uniform_(-self.init_w, self.init_w)

        self.fc2.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc2.bias.data.uniform_(-self.init_w, self.init_w)

        self.temperature = temperature
        self.use_softmax = use_softmax
        self.encode_feature_points = encode_feature_points

    def encode(self, input):
        h = self.encoder(input)

        if self.use_softmax:
            h = torch.exp(h / self.temperature)
            # sum over x, then sum over y
            total = h.sum(2).sum(2).view(h.shape[0], h.shape[1], 1, 1)
            h = h / total

        maps_x = torch.sum(h, 2)
        maps_y = torch.sum(h, 3)
        weights = ptu.from_numpy(np.arange(maps_x.shape[-1]) / maps_x.shape[-1])
        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        h = torch.cat([fp_x, fp_y], 1)

        mu = self.fc1(h) if self.encode_feature_points else h

        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return (mu, logvar)
