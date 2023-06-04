from __future__ import print_function

import numpy as np
import torch
from torch import nn

from rlkit.torch.networks import Flatten, Reshape
from rlkit.torch.networks.mlp import MultiHeadedMlp

from rlkit.torch.vae.vq_vae import Encoder, Decoder


def get_fancy_autoencoder_cnns(
            embedding_dim=3,
            input_channels=3,
            num_hiddens=128,
            num_residual_layers=3,
            num_residual_hiddens=64,
):
    encoder_cnn = Encoder(
        input_channels,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    )
    pre_rep_conv = nn.Conv2d(
        in_channels=num_hiddens,
        out_channels=embedding_dim,
        kernel_size=1,
        stride=1,
    )
    encoder_network = nn.Sequential(encoder_cnn, pre_rep_conv)
    decoder_network = Decoder(
        embedding_dim,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
    )
    return encoder_network, decoder_network


def get_fancy_vae(img_height, img_num_channels, img_width, latent_dim):
    encoder_cnn, decoder_cnn = get_fancy_autoencoder_cnns()
    # stub_vae = SashaVAE(latent_dim)
    # encoder_cnn = nn.Sequential(stub_vae._encoder, stub_vae._pre_rep_conv, )
    encoder_cnn.eval()
    test_mat = torch.zeros(1, img_num_channels, img_width, img_height, )
    encoder_cnn_output_shape = encoder_cnn(test_mat).shape[1:]
    encoder_cnn.train()
    encoder_cnn_output_size = np.prod(encoder_cnn_output_shape)
    encoder_mlp = MultiHeadedMlp(
        input_size=encoder_cnn_output_size,
        output_sizes=[latent_dim, latent_dim],
        hidden_sizes=[],
    )
    encoder_network = nn.Sequential(encoder_cnn, Flatten(), encoder_mlp)
    encoder_network.input_size = img_width * img_height * img_num_channels

    # decoder_cnn = stub_vae._decoder
    decoder_mlp = nn.Linear(latent_dim, encoder_cnn_output_size)
    decoder_network = nn.Sequential(
        decoder_mlp, Reshape(*encoder_cnn_output_shape), decoder_cnn,
    )
    decoder_network.input_size = encoder_cnn_output_size
    return decoder_network, encoder_network