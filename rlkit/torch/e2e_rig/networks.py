from rlkit.torch.core import PyTorchModule
from rlkit.torch.vae.vae_base import GaussianLatentVAE

imsize8_default_architecture=dict(
    conv_args=dict(
        kernel_sizes=[3],
        n_channels=[64],
        strides=[1],
    ),
    conv_kwargs=dict(
        hidden_sizes=[32, 32],
        conv_normalization_type="none",
        fc_normalization_type="none",
    ),
    deconv_args=dict(
        hidden_sizes=[],

        deconv_input_width=3,
        deconv_input_height=3,
        deconv_input_channels=64,

        deconv_output_kernel_size=6,
        deconv_output_strides=3,
        deconv_output_channels=3,

        kernel_sizes=[3],
        n_channels=[64],
        strides=[1],
    ),
    deconv_kwargs=dict(
        deconv_normalization_type="none",
        fc_normalization_type="none",
    )
)

class Vae2Encoder(PyTorchModule):
    def __init__(self, vae: GaussianLatentVAE):
        super(Vae2Encoder, self).__init__()
        self._vae = vae

    def forward(self, *input):
        params = self._vae.encode(*input)
        return params[0]