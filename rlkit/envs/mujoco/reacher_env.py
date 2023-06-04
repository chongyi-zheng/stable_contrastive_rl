from collections import OrderedDict

from gym.envs.mujoco.reacher import ReacherEnv as GymReacherEnv

from rlkit.core import logger as default_logger
from rlkit.core.eval_util import get_stat_in_paths, create_stats_ordered_dict


class ReacherEnv(GymReacherEnv):
    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('reward_dist', 'Distance Reward'),
            ('reward_ctrl', 'Action Reward'),
        ]:
            stat = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stat,
            ))
        distances = get_stat_in_paths(paths, 'env_infos', 'reward_dist')
        statistics.update(create_stats_ordered_dict(
            "Final Distance Reward",
            [ds[-1] for ds in distances],
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)


