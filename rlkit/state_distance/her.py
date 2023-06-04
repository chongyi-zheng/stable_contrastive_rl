from rlkit.state_distance.tdm_ddpg import TdmDdpg
from rlkit.state_distance.tdm_networks import TdmQf, TdmPolicy


class HER(TdmDdpg):
    """
    Hindsight Experience Replay
    """
    def __init__(
            self,
            env,
            qf,
            exploration_policy,
            ddpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=None,
            replay_buffer=None,
    ):
        tdm_kwargs.update(**dict(
            sample_rollout_goals_from='environment',
            sample_train_goals_from='her',
            vectorized=False,
            cycle_taus_for_rollout=False,
            max_tau=0,
            finite_horizon=False,
            dense_rewards=True,
            reward_type='indicator',
        ))
        if isinstance(qf, TdmQf):
            assert qf.structure == 'none'
        TdmDdpg.__init__(
            self,
            env,
            qf,
            exploration_policy,
            ddpg_kwargs,
            tdm_kwargs,
            base_kwargs,
            policy=policy,
            replay_buffer=replay_buffer,
        )


class HerQFunction(TdmQf):
    def __init__(
            self,
            env,
            **kwargs
    ):
        super().__init__(
            env,
            False,
            1,
            structure='none',
            **kwargs
        )


class HerPolicy(TdmPolicy):
    pass
