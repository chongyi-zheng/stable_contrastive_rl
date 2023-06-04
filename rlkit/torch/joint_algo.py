from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm

class JointAlgo(TorchRLAlgorithm):

    def __init__(
        self,
        algo1,
        algo2,
        algo1_prefix="Control_",
        algo2_prefix="Exploration_",
        secondary_rewards_name="exploration_rewards",
        *args,
        **kwargs
    ):
        self.algo1 = algo1
        self.algo2 = algo2
        self.algo1_prefix = algo1_prefix
        self.algo2_prefix = algo2_prefix
        self.secondary_rewards_name = secondary_rewards_name
        super().__init__(eval_policy=self.algo1.policy, *args, **kwargs)

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        exploration_rewards = batch[self.secondary_rewards_name]
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        self.algo1.need_to_update_eval_statistics = self.need_to_update_eval_statistics
        self.algo2.need_to_update_eval_statistics = self.need_to_update_eval_statistics

        self.algo1._train_given_data(
            rewards,
            terminals,
            obs,
            actions,
            next_obs,
            logger_prefix=self.algo1_prefix,
        )
        self.algo2._train_given_data(
            exploration_rewards,
            terminals,
            obs,
            actions,
            next_obs,
            logger_prefix=self.algo2_prefix,
        )
        if self.need_to_update_eval_statistics:
            self.eval_statistics = {
                **self.eval_statistics,
                **self.algo1.eval_statistics,
                **self.algo2.eval_statistics,
            }
            self.need_to_update_eval_statistics = False

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.algo1.update_epoch_snapshot(snapshot)
        self.algo2.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        self.algo1.update_epoch_snapshot(snapshot)
        self.algo2.update_epoch_snapshot(snapshot)

    @property
    def networks(self):
        return self.algo1.networks + self.algo2.networks

    @property
    def exploration_policy(self):
        if self.epoch % 2 == 0:
            return self.algo1.exploration_policy
        return self.algo2.exploration_policy

    @exploration_policy.setter
    def exploration_policy(self, val):
        # exploration policies are defined by the wrapped algorithms
        pass
