from rlkit.data_management.obs_dict_replay_buffer import \
    ObsDictRelabelingBuffer
from rlkit.torch.her.her import HER
from rlkit.torch.joint_algo import JointAlgo

class HerJointAlgo(HER, JointAlgo):
    def __init__(
            self,
            *args,
            policy=None,
            observation_key=None,
            desired_goal_key=None,
            **kwargs
    ):
        HER.__init__(
            self,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        self.policy = policy
        JointAlgo.__init__(self, *args, **kwargs)
        assert isinstance(
            self.replay_buffer, ObsDictRelabelingBuffer
        )
