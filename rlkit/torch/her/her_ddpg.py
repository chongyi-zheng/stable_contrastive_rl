from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.her.her import HER


class HerDdpg(HER, DDPG):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
