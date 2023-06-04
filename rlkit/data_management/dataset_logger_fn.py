"""Creates a callable that can be passed as a post_epoch_fn to an algorithm
for evaluating on specific data."""

from rlkit.core.logging import add_prefix

from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu

import numpy as np

class DatasetLoggerFn:
    def __init__(self, dataset, fn, prefix="", batch_size=64, *args, **kwargs):
        self.dataset = dataset
        self.fn = fn
        self.prefix = prefix
        self.batch_size = batch_size
        self.args = args
        self.kwargs = kwargs

    def __call__(self, algo):
        batch = self.dataset.random_batch(self.batch_size)
        batch = np_to_pytorch_batch(batch)
        log_dict = self.fn(batch, *self.args, **self.kwargs)
        return add_prefix(log_dict, self.prefix)

def run_bc_batch(batch, policy):
    o = batch["observations"]
    u = batch["actions"]
    # g = batch["resampled_goals"]
    # og = torch.cat((o, g), dim=1)
    og = o
    # pred_u, *_ = self.policy(og)
    dist = policy(og)
    pred_u, log_pi = dist.rsample_and_logprob()
    stats = dist.get_diagnostics()

    mse = (pred_u - u) ** 2
    mse_loss = np.mean(ptu.get_numpy(mse.mean()))

    policy_logpp = dist.log_prob(u, )
    logp_loss = -policy_logpp.mean()
    policy_loss = np.mean(ptu.get_numpy(logp_loss))

    return dict(
        bc_loss=policy_loss,
        mse_loss=mse_loss,
        **stats
    )