from rlkit.torch.sac.policies import *

class OneHotTauTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + max_tau + 1,
            output_size=output_size,
            **kwargs
        )
        self.max_tau = max_tau

    def forward(self, flat_obs, action):
        obs, taus = split_tau(flat_obs)
        h = torch.cat((obs, action), dim=1)

        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)

        h = torch.cat((
            obs,
            ptu.Variable(y_binary),
            action
        ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))
