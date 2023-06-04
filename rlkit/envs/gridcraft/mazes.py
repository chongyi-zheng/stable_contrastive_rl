from rlkit.envs.gridcraft.grid_spec import spec_from_string, \
    spec_from_sparse_locations, local_spec, \
    START, REWARD, REWARD2, REWARD3


MAZE_ARENA_64 = spec_from_sparse_locations(64, 64, {START: [(32,32)],
                                                 REWARD: [(0,0), (0,63), (63,0), (63,63)]})

MAZE_ARENA_32 = spec_from_sparse_locations(32, 32, {START: [(16,16)],
                                                    REWARD: [(0,0), (0,31), (31,0), (31,31)]})

MAZE_ARENA_16 = spec_from_sparse_locations(16, 16, {START: [(8,8)],
                                                    REWARD: [(0,0), (0,15), (15,0), (15,15)]})

REW_ARENA_64 = spec_from_sparse_locations(64, 64, {START: [(32,32)],
                                                    REWARD: [(4,4)],
                                                     REWARD2: local_spec(xpnt=(4,4), map="yyy\\"+
                                                                                           "yxy\\"+
                                                                                           "yyy"),
                                                     REWARD3: local_spec(xpnt=(4,4), map="yyyyy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yOxOy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yyyyy\\"),
                                                     })

REW_ARENA_128 = spec_from_sparse_locations(128, 128, {START: [(64,64)],
                                                     REWARD: [(10,10)],
                                                     REWARD2: local_spec(xpnt=(10,10), map="yyy\\"+
                                                                                           "yxy\\"+
                                                                                           "yyy"),
                                                     REWARD3: local_spec(xpnt=(10,10), map="yyyyy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yOxOy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yyyyy\\"),
                                                     })

MAZE1 = spec_from_string("SOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\\"+
                         "OO#OO#O##OOOOOOO#OOOOOOO#OOOORR\\"
                         )
MAZE2 = spec_from_string("S#OOO#OOO\\"+
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "O#O#O#O#O\\" +
                         "OOO#OOO#R\\"
                         )


MAZE3 = spec_from_string("SOOOOO\\"+
                         "OOOOOO\\"+
                         "OOOOOO\\"+
                         "OOOOOR\\"
                         )

MAZE4 = spec_from_string("S#OOO#OOOO\\"+
                         "OOO#OOO#O#\\" +
                         "##OOOOO#OO\\" +
                         "O#O#O#O##O\\" +
                         "O#O#O###OO\\" +
                         "OOO##ROOO#\\" +
                         "O#O#O#O#OO\\" +
                         "O#O#O#O#OO\\" +
                         "OOO#OOO#OO\\"
                         )

# Curriculum for Maze 5
MAZE5 = spec_from_string("SOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO\\"+
                         "OO#OO#O##OOOOOOO#OOOOOOO#OOOORR\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\")

MAZE5_1=spec_from_string("###############################\\"+
                         "########SS#####################\\"+
                         "########OO#####################\\"+
                         "########OO#####################\\"+
                         "########OO########ROOOOOOOOOOOS\\"+
                         "########OO#####################\\"+
                         "########OO#####################\\"+
                         "########RR#####################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\")


MAZE5_2=spec_from_string("###############################\\"+
                         "###############################\\"+
                         "#R#######R#####################\\"+
                         "#O#######O#####################\\"+
                         "#O#######O#####################\\"+
                         "#O#######O#####################\\"+
                         "#O#######O########SOOOOOOOOOR##\\"+
                         "#O#######S########SOOOOOOOOOR##\\"+
                         "#S#############################\\"+
                         "###############################\\"+
                         "###############################\\")

MAZE5_3=spec_from_string("###############################\\"+
                         "##############SOOOOOOOOOOOOR###\\"+
                         "###########################O###\\"+
                         "###########################O###\\"+
                         "###########################O###\\"+
                         "###########################O###\\"+
                         "###########################O###\\"+
                         "###########################S###\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\")

MAZE5_4=spec_from_string("###############################\\"+
                         "###############################\\"+
                         "######SOOOOOOOOOOOOORR#########\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############RROOOOOOOOOOOS##\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "###############################\\"+
                         "#SOOOOOOOOORR##################\\"+
                         "###############################\\")

MAZE6 = spec_from_string("###############################\\"+
                         "####OOOOOOOOOOOOOOOOOOOOOOOOO##\\"+
                         "####OOSOOOOOOOOOOOOOOOOOO333O##\\"+
                         "####OOOOOOOOOOOOOOOOOOOOOOO3O##\\"+
                         "####OO####################O3O##\\"+
                         "####OO############OOOOOOOOO2O##\\"+
                         "####OO############RRR2222222O##\\"+
                         "####OO####################OOO##\\"+
                         "####OOOOOOOO##############OOO##\\"+
                         "##########################OOO##\\"+
                         "###############################\\")

MAZE6NoReward = spec_from_string("###############################\\"+
                         "####OOOOOOOOOOOOOOOOOOOOOOOOO##\\"+
                         "####OOSOOOOOOOOOOOOOOOOOOOOOO##\\"+
                         "####OOOOOOOOOOOOOOOOOOOOOOOOO##\\"+
                         "####OO####################OOO##\\"+
                         "####OO############OOOOOOOOOOO##\\"+
                         "####OO############OOOOOOOOOOO##\\"+
                         "####OO####################OOO##\\"+
                         "####OOOOOOOO##############OOO##\\"+
                         "##########################OOO##\\"+
                         "###############################\\")

MAZE_ANY_START1 = spec_from_string("SSSSSSSSSS")
