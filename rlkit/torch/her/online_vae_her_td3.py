from rlkit.torch.her.her_td3 import HerTd3
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer

class OnlineVaeHerTd3(OnlineVaeAlgorithm, HerTd3):

    def __init__(
        self,
        online_vae_kwargs,
        base_kwargs,
        her_kwargs,
        td3_kwargs,
    ):
        OnlineVaeAlgorithm.__init__(
            self,
            **online_vae_kwargs,
        )
        HerTd3.__init__(
            self,
            base_kwargs=base_kwargs,
            td3_kwargs=td3_kwargs,
            her_kwargs=her_kwargs
        )

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)

    @property
    def networks(self):
        return HerTd3.networks.fget(self) + \
               OnlineVaeAlgorithm.networks.fget(self)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        HerTd3.update_epoch_snapshot(self, snapshot)
        OnlineVaeAlgorithm.update_epoch_snapshot(self, snapshot)
        return snapshot

