# import sys
# import os
# import stat


from rlkit.launchers import launcher_util as lu


def run_variant(experiment, variant):
    launcher_config = variant.get("launcher_config")
    lu.run_experiment(
        experiment,
        variant=variant,
        **launcher_config,
    )
