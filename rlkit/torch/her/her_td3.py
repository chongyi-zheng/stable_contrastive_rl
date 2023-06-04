from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.her.her import HERTrainer
from rlkit.torch.td3.td3 import TD3


class HerTd3(HERTrainer, TD3):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HERTrainer.__init__(
            self,
            **her_kwargs,
        )
        TD3.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
