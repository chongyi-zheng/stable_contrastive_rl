from collections import OrderedDict

from gym.envs.mujoco import HopperEnv as GymHopperEnv

from rlkit.core import logger as default_logger
from rlkit.core.eval_util import get_stat_in_paths, create_stats_ordered_dict


class HopperEnv(GymHopperEnv):
    def _step(self, a):
        ob, reward, done, _ = super()._step(a)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        return ob, reward, done, {
            'posafter': posafter,
            'height': height,
            'angle': ang,
        }

    def log_diagnostics(self, paths, logger=default_logger):
        statistics = OrderedDict()
        for name_in_env_infos, name_to_log in [
            ('posafter', 'Position'),
            ('height', 'Height'),
            ('angle', 'Angle'),
        ]:
            stats = get_stat_in_paths(paths, 'env_infos', name_in_env_infos)
            statistics.update(create_stats_ordered_dict(
                name_to_log,
                stats,
            ))
            statistics.update(create_stats_ordered_dict(
                "Final " + name_to_log,
                [s[-1] for s in stats],
            ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

