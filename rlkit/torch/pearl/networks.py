import torch
from torch import nn
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import ConcatMlp
import numpy as np


class MlpEncoder(ConcatMlp):
    '''
    encode context via MLP
    '''
    def __init__(self, *args, use_ground_truth_context=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ground_truth_context = use_ground_truth_context

    def forward(self, context):
        if self.use_ground_truth_context:
            return context
        else:
            return super().forward(context)

    def reset(self, num_tasks=1):
        pass


class MlpDecoder(ConcatMlp):
    '''
    decoder context via MLP
    '''
    pass


class DummyMlpEncoder(MlpEncoder):
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return 0 * output
        # TODO: check if this caused issues
        # z_dim = output.shape[-1]
        # num_components = output.shape[-2]
        # # Make it so that after a soft-plus + product of Gaussians, it always
        # # is a unit Gaussian
        # return torch.cat((
        #         0 * output[..., :z_dim//2],
        #         np.log(np.exp(num_components) - 1) + 0 * output[..., z_dim//2:],
        #     ), dim=-1,
        # )


class RecurrentEncoder(ConcatMlp):
    '''
    encode context via recurrent network
    '''

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))

        # input should be (task, seq, feat) and hidden should be (task, 1, feat)

        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=1, batch_first=True)

    def forward(self, in_, return_preactivations=False):
        # expects inputs of dimension (task, seq, feat)
        task, seq, feat = in_.size()
        out = in_.view(task * seq, feat)

        # embed with MLP
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            out = self.hidden_activation(out)

        out = out.view(task, seq, -1)
        out, (hn, cn) = self.lstm(out, (self.hidden, torch.zeros(self.hidden.size()).to(ptu.device)))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)

