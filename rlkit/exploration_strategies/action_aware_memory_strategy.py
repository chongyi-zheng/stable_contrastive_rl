from rlkit.exploration_strategies.base import (
    ExplorationStrategy,
    RawExplorationStrategy,
)


class ActionAwareMemoryStrategy(ExplorationStrategy):
    """
    Get the action for the environment. Add noise. Then use that noisy
    version to get the action for the memory.
    """
    def __init__(
            self,
            env_strategy: RawExplorationStrategy,
            write_strategy: RawExplorationStrategy,
    ):
        self.env_strategy = env_strategy
        self.write_strategy = write_strategy

    def get_action(self, t, observation, policy, **kwargs):
        raw_env_action, agent_info = policy.get_environment_action(observation)
        env_action = self.env_strategy.get_action_from_raw_action(
            raw_env_action,
            t=t,
        )
        raw_write_action, agent_info2 = (
            policy.get_write_action(env_action, observation)
        )
        write_action = self.write_strategy.get_action_from_raw_action(
            raw_write_action,
            t=t,
        )
        agent_info.update(agent_info2)
        return (env_action, write_action), agent_info
