import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.samplers.rollout_functions import (
    create_rollout_function,
    multitask_rollout,
)
from rlkit.torch.her.her import HER
from rlkit.torch.sac.sac import TwinSAC


class HerTwinSAC(HER, TwinSAC):
    def __init__(
            self,
            *args,
            twin_sac_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TwinSAC.__init__(self, *args, **kwargs, **twin_sac_kwargs, **base_kwargs)
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

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        HerTwinSAC.update_epoch_snapshot(self, snapshot)
        return snapshot
