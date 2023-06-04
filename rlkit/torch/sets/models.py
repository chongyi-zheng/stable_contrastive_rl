import numpy as np
from torch import nn

from rlkit.launchers.experiments.disentanglement import (
    contextual_encoder_distance_launcher as cedl,
)
from rlkit.torch.core import PyTorchModule
from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.networks import (
    BasicCNN,
    Flatten,
    Mlp,
    ConcatMultiHeadedMlp,
    Reshape,
)
from rlkit.torch.networks import basic
from rlkit.torch.networks.dcnn import BasicDCNN
from rlkit.torch.networks.mlp import MultiHeadedMlp
from rlkit.torch.networks.stochastic.distribution_generator import (
    BernoulliGenerator,
    Gaussian,
    IndependentGenerator,
)
from rlkit.torch.vae.vae_torch_trainer import VAE
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.sets.fancy_vae_architecture import (
    get_fancy_vae,
)


class DummyNetwork(PyTorchModule):
    def __init__(self, *output_shapes):
        super().__init__()
        self._output_shapes = output_shapes
        # if len(output_shapes) == 1:
        #     self.output = ptu.zeros(output_shapes[0])
        # else:
        #     self.output = tuple(
        #         ptu.zeros(shape) for shape in output_shapes
        #     )

    def forward(self, input):
        # import ipdb; ipdb.set_trace()
        if len(self._output_shapes) == 1:
            return ptu.zeros((input.shape[0], *self._output_shapes[0]))
        else:
            return tuple(
                ptu.zeros((input.shape[0], *shape))
                for shape in self._output_shapes
            )
        # return self.output


def create_dummy_image_vae(
        img_chw,
        latent_dim,
        *args,
        **kwargs
) -> VAE:
    encoder_network = DummyNetwork((latent_dim,), (latent_dim,))
    decoder_network = DummyNetwork(img_chw)
    encoder = Gaussian(encoder_network)
    decoder = Gaussian(decoder_network, std=1, reinterpreted_batch_ndims=3)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    return VAE(encoder, decoder, prior)


def create_image_vae(
    img_chw,
    latent_dim,
    encoder_cnn_kwargs,
    encoder_mlp_kwargs,
    decoder_mlp_kwargs=None,
    decoder_dcnn_kwargs=None,
    use_mlp_decoder=False,
    decoder_distribution="bernoulli",
    use_fancy_architecture=False,
) -> VAE:
    img_num_channels, img_height, img_width = img_chw
    if use_fancy_architecture:
        decoder_network, encoder_network = get_fancy_vae(img_height,
                                                         img_num_channels,
                                                         img_width, latent_dim)
    else:
        encoder_network = create_image_encoder(
            img_chw, latent_dim, encoder_cnn_kwargs, encoder_mlp_kwargs,
        )
        if decoder_mlp_kwargs is None:
            decoder_mlp_kwargs = cedl.invert_encoder_mlp_params(
                encoder_mlp_kwargs
            )
        if use_mlp_decoder:
            decoder_network = create_mlp_image_decoder(
                img_chw,
                latent_dim,
                decoder_mlp_kwargs,
                two_headed=decoder_distribution == 'gaussian_learned_variance',
            )
        else:
            if decoder_distribution == "gaussian_learned_variance":
                raise NotImplementedError()
            pre_dcnn_chw = encoder_network._modules["0"].output_shape
            if decoder_dcnn_kwargs is None:
                decoder_dcnn_kwargs = cedl.invert_encoder_params(
                    encoder_cnn_kwargs, img_num_channels,
                )
            decoder_network = create_image_decoder(
                pre_dcnn_chw,
                latent_dim,
                decoder_dcnn_kwargs,
                decoder_mlp_kwargs,
            )
    encoder = Gaussian(encoder_network)
    encoder.input_size = encoder_network.input_size
    if decoder_distribution in {
        "gaussian_learned_global_scalar_variance",
        "gaussian_learned_global_image_variance",
        "gaussian_learned_variance",
    }:
        if decoder_distribution == "gaussian_learned_global_image_variance":
            log_std = basic.LearnedPositiveConstant(
                ptu.zeros((img_num_channels, img_height, img_width))
            )
            decoder_network = basic.ApplyMany(decoder_network, log_std)
        elif decoder_distribution == "gaussian_learned_global_scalar_variance":
            log_std = basic.LearnedPositiveConstant(ptu.zeros(1))
            decoder_network = basic.ApplyMany(decoder_network, log_std)
        decoder = Gaussian(decoder_network, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "gaussian_fixed_unit_variance":
        decoder = Gaussian(decoder_network, std=1, reinterpreted_batch_ndims=3)
    elif decoder_distribution == "bernoulli":
        decoder = IndependentGenerator(
            BernoulliGenerator(decoder_network), reinterpreted_batch_ndims=3
        )
    else:
        raise NotImplementedError(decoder_distribution)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    return VAE(encoder, decoder, prior)


def create_image_encoder(
    img_chw, latent_dim, encoder_cnn_kwargs, encoder_kwargs,
):
    img_num_channels, img_height, img_width = img_chw
    cnn = BasicCNN(
        input_width=img_width,
        input_height=img_height,
        input_channels=img_num_channels,
        **encoder_cnn_kwargs
    )
    cnn_output_size = np.prod(cnn.output_shape)
    mlp = MultiHeadedMlp(
        input_size=cnn_output_size,
        output_sizes=[latent_dim, latent_dim],
        **encoder_kwargs
    )
    enc = nn.Sequential(cnn, Flatten(), mlp)
    enc.input_size = img_width * img_height * img_num_channels
    enc.output_size = latent_dim
    return enc


def create_image_decoder(
    pre_dcnn_chw,
    latent_dim,
    decoder_dcnn_kwargs,
    decoder_kwargs,
):
    dcnn_in_channels, dcnn_in_height, dcnn_in_width = pre_dcnn_chw
    dcnn_input_size = dcnn_in_channels * dcnn_in_width * dcnn_in_height
    dcnn = BasicDCNN(
        input_width=dcnn_in_width,
        input_height=dcnn_in_height,
        input_channels=dcnn_in_channels,
        **decoder_dcnn_kwargs
    )
    mlp = Mlp(
        input_size=latent_dim, output_size=dcnn_input_size, **decoder_kwargs
    )
    dec = nn.Sequential(mlp, dcnn)
    dec.input_size = latent_dim
    return dec


def create_mlp_image_decoder(
    img_chw, latent_dim, decoder_kwargs, two_headed,
):
    img_num_channels, img_height, img_width = img_chw
    output_size = img_num_channels * img_height * img_width
    if two_headed:
        dec = nn.Sequential(
            MultiHeadedMlp(
                input_size=latent_dim,
                output_sizes=[output_size, output_size],
                **decoder_kwargs
            ),
            basic.Map(Reshape(img_num_channels, img_height, img_width)),
        )
    else:
        dec = nn.Sequential(
            Mlp(
                input_size=latent_dim, output_size=output_size, **decoder_kwargs
            ),
            Reshape(img_num_channels, img_height, img_width),
        )
    dec.input_size = latent_dim
    dec.output_size = img_num_channels * img_height * img_width
    return dec


def create_vector_vae(data_dim, latent_dim, encoder_kwargs):
    encoder = create_vector_encoder(data_dim, latent_dim, encoder_kwargs)
    decoder_kwargs = cedl.invert_encoder_mlp_params(encoder_kwargs)
    decoder = create_vector_decoder(data_dim, latent_dim, decoder_kwargs)
    prior = MultivariateDiagonalNormal(
        loc=ptu.zeros(1, latent_dim), scale_diag=ptu.ones(1, latent_dim),
    )
    return VAE(encoder, decoder, prior)


def create_vector_encoder(data_dim, latent_dim, encoder_kwargs):
    enc = ConcatMultiHeadedMlp(
        input_size=data_dim,
        output_sizes=[latent_dim, latent_dim],
        **encoder_kwargs
    )
    enc.input_size = data_dim
    enc.output_size = latent_dim
    return enc


def create_vector_decoder(data_dim, latent_dim, decoder_kwargs):
    dec = Mlp(input_size=latent_dim, output_size=data_dim, **decoder_kwargs)
    return dec
