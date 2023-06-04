from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.her.her_exploration import HERExploration
from rlkit.torch.td3.td3_ensemble_qs import TD3EnsembleQs


class HerExplorationTd3(HERExploration, TD3EnsembleQs):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HERExploration.__init__(
            self,
            **her_kwargs,
        )
        TD3EnsembleQs.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
