import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.samplers.rollout_functions import (
    create_rollout_function,
    multitask_rollout,
)
from rlkit.torch.her.her import HER
from rlkit.torch.sac.sac import SoftActorCritic


class HerSac(HER, SoftActorCritic):
    def __init__(
            self,
            *args,
            observation_key=None,
            desired_goal_key=None,
            **kwargs
    ):
        HER.__init__(
            self,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        SoftActorCritic.__init__(self, *args, **kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )

    @property
    def eval_rollout_function(self):
        return create_rollout_function(
            multitask_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key,
        )
