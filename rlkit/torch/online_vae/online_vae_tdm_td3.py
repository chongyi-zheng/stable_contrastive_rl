from rlkit.state_distance.tdm_td3 import TdmTd3
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer

class OnlineVaeTdmTd3(OnlineVaeAlgorithm, TdmTd3):

    def __init__(
        self,
        tdm_td3_kwargs,
        online_vae_kwargs,
    ):
        OnlineVaeAlgorithm.__init__(self, **online_vae_kwargs)
        TdmTd3.__init__(self, **tdm_td3_kwargs)

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)

    @property
    def networks(self):
        return TdmTd3.networks.fget(self) + \
               OnlineVaeAlgorithm.networks.fget(self)
