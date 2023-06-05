drawer_pnp_push_commands = [
    ### Task 0 ###
    {
        "drawer_open": False,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top_obj",
                }
             )
        ]
    },
    ### Task 1 ###
    {
        "drawer_open": False,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ]
    },
    ### Task 2 ###
    {
        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {})
        ]
    },
    ### Task 3 ###
    {
        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ]
    },
    ##### More PNP Tasks #####
    ### Task 4 ###
    {
        "drawer_open": False,
        "drawer_yaw": 99.18182653164543,
        "drawer_quadrant": 1,
        "small_object_pos": [0.5125003, -0.16741399, -0.24],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "out",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             )
        ]
    },
    ### Task 5 ###
    {
        "drawer_open": True,
        "drawer_yaw": 99.18182653164543,
        "drawer_quadrant": 1,
        "small_object_pos": [0.5125003, -0.16741399, -0.24],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "in",
                }
             )
        ]
    },
    ### Task 6 ###
    {
        "drawer_open": False,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.7480996, -0.08887033, -0.27],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             )
        ]
    },
    ### Task 7 ###
    {
        "drawer_open": True,
        "drawer_yaw": 171.60405411742914,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52314476, -0.17986905, -0.27851065],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "in",
                }
             )
        ]
    },

    ##### V4 PNP #####
    ### Task 8 ###
    {
        "drawer_open": True,
        "drawer_yaw": 132.41364075031314,
        "drawer_quadrant": 1,
        "small_object_pos": [0.56538715,  0.13505916, -0.35201065],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             )
        ]
    },
    ### Task 9 ###
    {
        "drawer_open": False,
        "drawer_yaw": 174.04400595864604,
        "drawer_quadrant": 1,
        "small_object_pos": [0.51182607, -0.16842074, -0.27851074],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "out",
                    "target_position": [0.73242832, 0.1195218, -0.35201056],
                }
             )
        ]
    },
    ##### N-Stage Tasks #####

    #### 1-Stage Tasks ####
    ### Task 10 ###
    {
        "drawer_open": False,
        "drawer_yaw": 148.969115489417,
        "drawer_quadrant": 1,
        "small_object_pos": [0.56538715,  0.13505916, -0.35201065],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
        ]
    },
    ### Task 11 ###
    {
        "drawer_open": False,
        "drawer_yaw": 148.969115489417,
        "drawer_quadrant": 1,
        "small_object_pos": [0.56538715,  0.13505916, -0.35201065],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
        ]
    },
    ### Task 12 ###
    {
        "drawer_open": False,
        "drawer_yaw": 148.969115489417,
        "drawer_quadrant": 1,
        "small_object_pos": [0.56538715,  0.13505916, -0.35201065],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             )
        ]
    },

    #### 2-Stage Tasks ####
    ### Task 13 ###
    {
        "init_pos": [0.6658003099428647, 0.06904850095001731, -0.11053177166415183],
        "drawer_open": True,
        "drawer_yaw": 148.969115489417,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
        ]
    },
    ### Task 14 ###
    {
        # "init_pos": [0.6, 0.2, -0.1],
        # "init_pos": [0.7977056203444699, 0.1630329037099802, -0.10748435590195876],

        "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        "init_theta": [180, 0, 90],

        # "init_pos": [0.7916068250737145, 0.16476240265540493, -0.17648835260509743],
        # "init_theta": [-178.281349, 0.127151952, 85.3789757],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ],
        "no_collision_handle_and_cylinder": True,
    },
    ### Task 15 ###
    {
        "drawer_open": False,
        "drawer_yaw": 21.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.72105773, 0.03390593, -0.35201112],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
        ]
    },
    ### Task 16 ###
    {
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ]
    },


    #### 3-Stage Tasks ####
    ### Task 17 ###
    {
        "drawer_open": True,
        "drawer_yaw": 106.07707109729891,
        "drawer_quadrant": 1,
        "small_object_pos": [0.55574489, 0.14969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
        ]
    },
    ### Task 18 ###
    {
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 0,
                }
             ),
        ]
    },
    ### Task 19 ###
    {
        "drawer_open": False,
        "drawer_yaw": 21.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.72105773, 0.03390593, -0.35201112],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
            ("move_obj_pnp",
                {
                    "target_location": "in",
                }
             ),
        ]
    },
    ### Task 20 ###
    {
        "drawer_open": True,
        "drawer_yaw": 95.41568591295118,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52499483, -0.16752302, -0.27851613],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 0,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
        ]
    },

    ### Task 21 ###
    {
        # "init_pos": [0.7752032801275157, 0.07302986833527307, -0.10714898300234312],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": False,
        "drawer_yaw": 158.969115489417,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 0,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 22 ###
    {
        "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "out",
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 23 ###
    {
        # "init_pos": [0.6529509743363572, -0.1566087618206431, -0.11210102868603561],
        # "init_theta": [-179., 0., 0.],
        # "init_pos": [0.6485792609689918, -0.15005746455198332, -0.15384601238446874],
        # "init_theta": [-179., 0., -82.],
        "drawer_open": True,
        "drawer_yaw": 95.41568591295118,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52499483, -0.16752302, -0.27851613],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "in",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             ),
            ("move_obj_slide",
                {
                    "target_quadrant": 0,
                }
             )
        ]
    },

    ### Task 24 ###
    {
        # "init_pos": [0.6529509743363572, -0.1566087618206431, -0.11210102868603561],
        # "init_theta": [-179., 0., 0.],
        "init_pos": [0.6485792609689918, -0.15005746455198332, -0.15384601238446874],
        # "init_theta": [-179., 0., -82.],
        "drawer_open": False,
        "drawer_yaw": 95.41568591295118,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52499483, -0.16752302, -0.27851613],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_pnp",
                {
                    "target_location": "in",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             )
        ]
    },

    ### Task 25 ###
    {
        "drawer_open": False,
        "drawer_yaw": 51.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_pnp",
                {
                    "target_location": "in",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             )
        ]
    },

    ### Task 26 ###
    {
        # "init_pos": [0.6, 0.2, -0.1],
        # "init_pos": [0.7977056203444699, 0.1630329037099802, -0.10748435590195876],

        # "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        # "init_theta": [-178.281349, 0.127151952, 15.3789757],

        # "init_pos": [0.7916068250737145, 0.16476240265540493, -0.17648835260509743],
        # "init_theta": [-178.281349, 0.127151952, 85.3789757],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "in",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             ),
            ("move_obj_slide",
                {
                    "target_quadrant": 1,
                }
             )
        ]
    },


    ### Task 27 ###
    {
        # "init_pos": [0.6, 0.2, -0.1],
        # "init_pos": [0.7977056203444699, 0.1630329037099802, -0.10748435590195876],

        # "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        # "init_theta": [-178.281349, 0.127151952, 15.3789757],

        # "init_pos": [0.7916068250737145, 0.16476240265540493, -0.17648835260509743],
        # "init_theta": [-178.281349, 0.127151952, 85.3789757],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos":  # [0.52499483, -0.16752302, -0.27851613],
        [0.72105773, 0.03390593, -0.35201112],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "in",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             ),
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             )
        ]
    },

    ### Task 28 ###
    {
        "drawer_open": False,
        "drawer_yaw": 21.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
        ]
    },


    ### Task 29 ###
    {
        "drawer_open": False,
        "drawer_yaw": 21.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.72105773, 0.03390593, -0.35201112],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 30 ###
    {

        "init_pos": [0.705514331327057, 0.0834841537873687, -0.15236609064788176],
        "drawer_open": False,
        "drawer_yaw": 5.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.73105773, 0.09390593, -0.35201112],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                    "target_position": [0.50419498, 0.14887659, -0.34],
                }
             ),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
        ]
    },

    ### Task 31 ###
    {
        "init_pos": [0.5024816134266682, -0.07439965024142557, -0.17519156942543912],
        "drawer_open": False,
        "drawer_yaw": 11.04125081594207,
        "drawer_quadrant": 0,
        "small_object_pos": [0.70574489, 0.22969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 1,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             ),
            ("move_drawer", {}),
        ],
        "drawer_hack": True,
    },

    ### Task 32 ###
    {
        "drawer_open": False,
        # "drawer_yaw": 95.41568591295118,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 33 ###
    {
        "drawer_open": False,
        # "drawer_yaw": 95.41568591295118,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 0,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 34 ###
    {

        "init_pos": [0.8301508583634503, 0.06666064578268723, -0.11697392914044442],
        "drawer_open": False,
        "drawer_yaw": 115.,
        "drawer_quadrant": 1,
        "small_object_pos": [0.50574489, 0.16969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 0,
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 35 ###
    {
        "init_pos": [0.8301508583634503, 0.06666064578268723, -0.11697392914044442],
        "drawer_open": False,
        "drawer_yaw": 130.,
        "drawer_quadrant": 1,
        "small_object_pos": [0.50574489, 0.16969248, -0.35201094],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_slide",
                {
                    "target_quadrant": 0,
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 36 ###
    {
        "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 37 ###
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ],
    },

    ### Task 38 (Harder Task 14) ###
    {
        "init_theta": [180, 0, 90],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 3,
                }
             )
        ],
        "no_collision_handle_and_cylinder": True,
    },

    ### Task 39 (a place holder) ###
    {
        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
        ]
    },

    ### Task 40 (Single-step version of Task 14) ###
    {
        # "init_pos": [0.6, 0.2, -0.1],
        # "init_pos": [0.7977056203444699, 0.1630329037099802, -0.10748435590195876],

        "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        "init_theta": [180, 0, 90],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
        ],
        "no_collision_handle_and_cylinder": True,
    },

    ### Task 41 (Single-step version of Task 14) ###
    {
        # "init_pos": [0.6, 0.2, -0.1],
        # "init_pos": [0.7977056203444699, 0.1630329037099802, -0.10748435590195876],

        "init_pos": [0.8007752866589611, 0.1648154585720787, -0.14012390258012603],
        "init_theta": [180, 0, 90],

        "drawer_open": True,
        "drawer_yaw": 89.60575853084282,
        "drawer_quadrant": 0,
        "small_object_pos": [0.51250274, 0.16758655, -0.26951355],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_drawer", {}),
        ],
        "no_collision_handle_and_cylinder": True,
    },

    ### Task 42 (Task 37 on table) ###
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "out",
                    "target_position": [0.75, -0.14, -0.34],
                }
             ),
            ("move_drawer", {}),
        ]
    },

    ### Task 43 (Task 34 with cylinder on left) ###
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ],
    },

    ### Task 44 (Task 34 flipped with cylinder on right) ###
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, .05 + -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 180-171.86987153482346,
        "drawer_quadrant": 0,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ],
    },

    ### Task 45 (Task 34 flipped with cylinder on left) ###
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, .05 + -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 180-171.86987153482346,
        "drawer_quadrant": 0,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "top",
                }
             ),
            ("move_drawer", {}),
        ],
    },

    ### Task 46
    {
        # "init_pos": [0.5634446177134209, -0.02899667408883611, -0.228462328752362],
        "init_pos": [0.5559647343004194, -0.027816898388712933, -0.28124351873499637],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.54798087, -0.006632, -0.34451221],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 2,
        "command_sequence": [
            ("move_obj_pnp",
                {
                    "target_location": "out",
                }
             ),
            ("move_drawer", {}),
        ],
        "no_collision_handle_and_small": True,
    },

    ### Task 47
    {
        # "init_pos": [0.5454021011892545, -0.025902720975473824, -0.14055474048627775],
        # "init_pos": [0.5510826127903993, 0.12574368405456257, -0.10239721273123115],
        "init_pos": [0.5696635276889644, 0.10309521526108646, -0.1203028851440111],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 0,
                }
             )
        ],
        "no_collision_handle_and_cylinder": True,
    },

    ### Task 48
    {
        # "init_pos": [0.5454021011892545, -0.025902720975473824, -0.14055474048627775],
        # "init_pos": [0.5510826127903993, 0.12574368405456257, -0.10239721273123115],
        "init_pos": [0.5696635276889644, 0.10309521526108646, -0.1203028851440111],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_pnp",
                {
                    "target_location": "out",
                }
             ),
        ],
        "no_collision_handle_and_small": True,
    },

    ### Task 49
    {
        # "init_pos": [0.5454021011892545, -0.025902720975473824, -0.14055474048627775],
        # "init_pos": [0.5510826127903993, 0.12574368405456257, -0.10239721273123115],
        "init_pos": [0.5696635276889644, 0.10309521526108646, -0.1203028851440111],
        # "init_theta": [-179., 0., 0.],
        "drawer_open": True,
        "drawer_yaw": 171.86987153482346,
        "drawer_quadrant": 1,
        "small_object_pos": [0.52500062, -0.16750046, -0.27851104],
        "small_object_pos_randomness": {
            "low": [-0.00, -0.00, 0],
            "high": [0.00, 0.00, 0],
        },
        "large_object_quadrant": 3,
        "command_sequence": [
            ("move_drawer", {}),
            ("move_obj_slide",
                {
                    "target_quadrant": 2,
                }
             )
        ],
        "no_collision_handle_and_small": True,
    },
]
