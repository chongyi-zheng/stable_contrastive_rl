import numpy as np

from rlkit.policies.base import Policy
from rlkit.torch.naf.naf import NafPolicy


class CombinedNafPolicy(Policy):
    def __init__(
            self,
            policy1: NafPolicy,
            policy2: NafPolicy,
    ):
        self.policy1 = policy1
        self.policy2 = policy2

    def get_action(self, obs):
        mu1, P1 = self.policy1.get_action_and_P_matrix(obs)
        mu2, P2 = self.policy2.get_action_and_P_matrix(obs)
        inv = np.linalg.inv(P1 + P2)
        return inv @ (P1 @ mu1 + P2 @ mu2), {}

    def log_diagnostics(self, paths):
        pass


class AveragerPolicy(Policy):
    def __init__(self, policy1, policy2):
        self.policy1 = policy1
        self.policy2 = policy2

    def get_action(self, obs):
        action1, info_dict1 = self.policy1.get_action(obs)
        action2, info_dict2 = self.policy2.get_action(obs)
        return (action1 + action2) / 2, dict(info_dict1, **info_dict2)

    def log_diagnostics(self, paths):
        pass
