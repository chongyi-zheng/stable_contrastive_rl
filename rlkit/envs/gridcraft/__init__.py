from gym.envs.registration import register
import logging

from rlkit.envs.gridcraft.mazes import *

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_grid_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(id='GridMaze1-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE1})
    register(id='GridMaze2-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE2})
    register(id='GridMaze3-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE3})

    register(id='GridArena64-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': MAZE_ARENA_64})
    register(id='GridArena32-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': MAZE_ARENA_32})
    register(id='GridArena16-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': MAZE_ARENA_16})


    register(id='RewArena64-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': REW_ARENA_64})
    register(id='RewArena64FS10-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': REW_ARENA_64,
                     'frameskip':10})
    register(id='RewArena128-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': REW_ARENA_128})
    register(id='RewArena128FS10-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'coordinate_wise': True, 'gridspec': REW_ARENA_128,
                     'frameskip':10})

    #curriculum mazes
    register(id='GridMaze5-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE5})
    register(id='GridMaze5_1-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE5_1})
    register(id='GridMaze5_2-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE5_2})
    register(id='GridMaze5_3-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE5_3})
    register(id='GridMaze5_4-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'add_eyes': True, 'gridspec': MAZE5_4})

    register(id='GridMaze6-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'coordinate_wise': True, 'add_eyes': True, 'gridspec': MAZE6})
    register(id='GridMaze6NoReward-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={'one_hot': True, 'coordinate_wise': True, 'add_eyes': True, 'gridspec': MAZE6, 'zero_reward': True})

    # Mazes where you can start wherever
    register(id='GridMazeAnyStart1-v0',
             entry_point='railrl.envs.gridcraft.grid_env:GridEnv',
             kwargs={
                 'one_hot': True,
                 'coordinate_wise': True,
                 'add_eyes': False,
                 'gridspec': MAZE_ANY_START1,
                 'zero_reward': True,
             })


    LOGGER.info("Finished registering custom gym environments")
register_grid_envs()
