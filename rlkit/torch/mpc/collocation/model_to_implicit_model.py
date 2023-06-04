from rlkit.torch.core import PyTorchModule


class ModelToImplicitModel(PyTorchModule):
    def __init__(self, model, bias=0, order=2):
        super().__init__()
        self.model = model
        self.bias = bias
        self.order = order

    def forward(self, obs, action, next_obs):
        diff = next_obs - obs - self.model(obs, action)
        if self.order == 2:
            return -(diff**2).sum(dim=1, keepdim=True) + self.bias
        else:
            return -diff.abs().sum(dim=1, keepdim=True) + self.bias
