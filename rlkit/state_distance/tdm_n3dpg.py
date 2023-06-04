from rlkit.state_distance.tdm import TemporalDifferenceModel
from rlkit.torch.ddpg.n3dpg import N3DPG


class TdmN3dpg(TemporalDifferenceModel, N3DPG):
    def __init__(
            self,
            env,
            qf,
            vf,
            exploration_policy,
            n3dpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        N3DPG.__init__(
            self,
            env=env,
            qf=qf,
            vf=vf,
            policy=policy,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **n3dpg_kwargs,
            **base_kwargs
        )
        super().__init__(**tdm_kwargs)

    def _do_training(self):
        N3DPG._do_training(self)

    def evaluate(self, epoch):
        N3DPG.evaluate(self, epoch)
