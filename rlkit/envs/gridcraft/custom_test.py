from rlkit.envs.gridcraft import REW_ARENA_64
from rlkit.envs.gridcraft.grid_env import GridEnv
from rlkit.envs.gridcraft.grid_spec import *
from rlkit.envs.gridcraft.mazes import MAZE_ANY_START1
import gym.spaces.prng as prng
import numpy as np

if __name__ == "__main__":
    prng.seed(2)

    maze_spec = \
        spec_from_string("SOOOO#R#OO\\"+
                         "OSOOO#2##O\\" +
                         "###OO#3O#O\\" +
                         "OOOOO#OO#O\\" +
                         "OOOOOOOOOO\\"
                         )

    #maze_spec = spec_from_sparse_locations(50, 50, {START: [(25,25)], REWARD: [(45,45)]})
    # maze_spec = REW_ARENA_64
    maze_spec = MAZE_ANY_START1

    env = GridEnv(maze_spec, one_hot=True, add_eyes=True, coordinate_wise=True)

    s = env.reset()
    #env.render()

    obses = []
    for t in range(10):
        a = env.action_space.sample()
        obs, r, done, infos = env.step(a, verbose=True)
        obses.append(obs)
    obses = np.array(obses)

    paths = [{'observations': obses}]
    env.plot_trajs(paths)