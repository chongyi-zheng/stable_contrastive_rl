from rlkit.torch.her.her_twin_sac import HerTwinSAC
from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer

class OnlineVaeHerTwinSac(OnlineVaeAlgorithm, HerTwinSAC):

    def __init__(
        self,
        online_vae_kwargs,
        base_kwargs,
        her_kwargs,
        twin_sac_kwargs,
    ):
        OnlineVaeAlgorithm.__init__(
            self,
            **online_vae_kwargs,
        )
        HerTwinSAC.__init__(
            self,
            base_kwargs=base_kwargs,
            twin_sac_kwargs=twin_sac_kwargs,
            her_kwargs=her_kwargs
        )

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)

    @property
    def networks(self):
        return HerTwinSAC.networks.fget(self) + \
               OnlineVaeAlgorithm.networks.fget(self)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        OnlineVaeAlgorithm.update_epoch_snapshot(self, snapshot)
        return snapshot