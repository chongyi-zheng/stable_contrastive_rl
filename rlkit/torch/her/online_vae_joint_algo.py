from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.data_management.online_vae_replay_buffer \
        import OnlineVaeRelabelingBuffer
from rlkit.torch.her.her_joint_algo import HerJointAlgo

class OnlineVaeHerJointAlgo(OnlineVaeAlgorithm, HerJointAlgo):

    def __init__(
        self,
        vae,
        vae_trainer,
        *algo_args,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,
        oracle_data=False,

        **algo_kwargs
    ):

        OnlineVaeAlgorithm.__init__(
            self,
            vae,
            vae_trainer,
            vae_save_period=vae_save_period,
            vae_training_schedule=vae_training_schedule,
            oracle_data=oracle_data,
        )
        HerJointAlgo.__init__(self, *algo_args, **algo_kwargs)

        assert isinstance(self.replay_buffer, OnlineVaeRelabelingBuffer)

    @property
    def networks(self):
        return OnlineVaeAlgorithm.networks.fget(self) + \
               HerJointAlgo.networks.fget(self)

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        OnlineVaeAlgorithm.update_epoch_snapshot(self, snapshot)
        HerJointAlgo.update_epoch_snapshot(self, snapshot)
        return snapshot

