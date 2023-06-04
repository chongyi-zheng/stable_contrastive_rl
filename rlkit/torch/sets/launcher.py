import numpy as np
import torch
from torch.utils import data

from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.core import logger
from rlkit.envs.images import EnvRenderer
from rlkit.envs.pygame import pnp_util
from rlkit.util import ml_util
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.sets import set_projection
from rlkit.torch.sets.set_vae_trainer import SetVAETrainer
from rlkit.torch.sets.unsupervised_algorithm import (
    DictLoader,
    UnsupervisedTorchAlgorithm,
)
from rlkit.torch.sets.models import create_image_vae


def generate_images(
        env,
        env_renderer,
        num_images=32,
):
    for state in pnp_util.generate_goals(env, num_images):
        env._set_positions(state)
        img = env_renderer(env)
        yield img


def sample_axis_set_projector(max_index):
    index = np.random.randint(0, max_index//2)
    value = np.random.uniform(-4, 4, 1)
    return set_projection.ProjectOntoAxis({index: value})


def sample_point_set_projector(max_index):
    index = np.random.randint(0, max_index//2)
    value = np.random.uniform(-4, 4, 1)
    value2 = np.random.uniform(-4, 4, 1)
    return set_projection.ProjectOntoAxis({
        2*index: value,
        2*index+1: value2,
    })


def sample_set_projector():
    sampler = np.random.choice([
        sample_axis_set_projector,
        sample_point_set_projector,
    ])
    return sampler()


def create_pygame_env(num_objects):
    return PickAndPlaceEnv(
        # Environment dynamics
        action_scale=1.0,
        ball_radius=0.75,  # 1.
        boundary_dist=4,
        object_radius=0.50,
        min_grab_distance=0.5,
        walls=None,
        # Rewards
        action_l2norm_penalty=0,
        reward_type="dense",  # dense_l1
        success_threshold=0.60,
        # Reset settings
        fixed_goal=None,
        # Visualization settings
        images_are_rgb=True,
        render_dt_msec=0,
        render_onscreen=False,
        render_size=84,
        show_goal=False,
        # get_image_base_render_size=(48, 48),
        # Goal sampling
        goal_samplers=None,
        goal_sampling_mode='random',
        num_presampled_goals=10000,
        object_reward_only=True,
        init_position_strategy='random',
        num_objects=num_objects,
    )


def create_pybullet_env(num_objects):
    from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC
    env = SawyerLiftEnvGC(
        action_scale=.06,
        action_repeat=10, #5
        timestep=1./120, #1./240
        solver_iterations=500, #150
        max_force=1000,
        gui=False,
        num_obj=num_objects,
        pos_init=[.75, -.3, 0],
        pos_high=[.75, .4, .3],
        pos_low=[.75, -.4, -.36],
        reset_obj_in_hand_rate=0.0,
        goal_sampling_mode='ground',
        random_init_bowl_pos=False,
        sliding_bowl=False,
        heavy_bowl=False,
        bowl_bounds=[-0.40, 0.40],
        reward_type='obj_dist',
        use_rotated_gripper=True,  # False
        use_wide_gripper=True,  # False
        soft_clip=True,
        obj_urdf='spam',
        max_joint_velocity=None,
    )
    env.num_objects = num_objects
    return env


def create_env(version='pygame', num_objects=4):
    if version == 'pygame':
        return create_pygame_env(num_objects=num_objects)
    elif version == 'pybullet':
        return create_pybullet_env(num_objects=num_objects)
    else:
        raise NotImplementedError()


def generate_set_images(
        env,
        env_renderer,
        num_sets=1,
        num_samples_per_set=32,
):

    set_descriptions = []
    for _ in range(num_sets // 2):
        set_descriptions.append(sample_point_set_projector(env.num_objects*2))
        set_descriptions.append(sample_axis_set_projector(env.num_objects*2))
    if num_sets % 2 == 1:
        set_descriptions.append(sample_point_set_projector(env.num_objects*2))

    # set_descriptions = [sample_set_projector() for _ in range(num_sets)]
    sets = pnp_util.sample_sets(env, set_descriptions, n=num_samples_per_set)

    def create_images(states):
        for state in states:
            env._set_positions(state)
            img = env_renderer(env)
            yield img

    for states in sets:
        yield list(create_images(states))


def save_images(images):
    from moviepy import editor as mpy
    def create_video(imgs):
        imgs = np.array(imgs).transpose([0, 2, 3, 1])
        imgs = (255 * imgs).astype(np.uint8)
        return mpy.ImageSequenceClip(list(imgs), fps=5)

    def concatenate_imgs_into_video(images_list):
        subclips = [create_video(imgs) for imgs in images_list]
        together = mpy.clips_array([subclips])
        together.write_videofile('/home/vitchyr/tmp.mp4')

    concatenate_imgs_into_video(images)


def infinite(iterator):
    while True:
        for x in iterator:
            yield x


def create_beta_schedule(version='none', x_values=None, y_values=None):
    if version == 'none':
        return None
    elif version == 'piecewise_linear':
        return ml_util.PiecewiseLinearSchedule(
            x_values,
            y_values,
        )
    else:
        raise NotImplementedError()



def test_offline_set_vae(
        create_vae_kwargs,
        vae_trainer_kwargs,
        algo_kwargs,
        data_loader_kwargs,
        generate_set_kwargs,
        num_ungrouped_images,
        env_kwargs,
        beta_schedule_kwargs=None,
):
    if beta_schedule_kwargs is None:
        beta_schedule_kwargs = {}
    eval_set_imgs, renderer, set_imgs, set_imgs_iterator, ungrouped_imgs = create_dataset(
        env_kwargs, generate_set_kwargs, num_ungrouped_images)

    set_imgs_flat = set_imgs.view((-1, *set_imgs.shape[-3:]))
    all_imgs = torch.cat([ungrouped_imgs, set_imgs_flat], dim=0)
    all_imgs_iterator = data.DataLoader(all_imgs, **data_loader_kwargs)

    vae = create_image_vae(
        img_chw=renderer.image_chw,
        **create_vae_kwargs
    )

    set_key = 'set'
    data_key = 'data'
    dict_loader = DictLoader({
        data_key: all_imgs_iterator,
        set_key: infinite(set_imgs_iterator),
    })
    beta_schedule = create_beta_schedule(**beta_schedule_kwargs)
    vae_trainer = SetVAETrainer(
        vae=vae,
        set_key=set_key,
        data_key=data_key,
        train_sets=set_imgs[:2],
        eval_sets=eval_set_imgs[:2],
        beta_schedule=beta_schedule,
        **vae_trainer_kwargs)
    algorithm = UnsupervisedTorchAlgorithm(
        vae_trainer,
        dict_loader,
        **algo_kwargs,
    )
    algorithm.to(ptu.device)
    algorithm.run()
    print(logger.get_snapshot_dir())


def create_dataset(env_kwargs, generate_set_kwargs, num_ungrouped_images):
    env = create_env(**env_kwargs)
    renderer = EnvRenderer(output_image_format='CHW')
    import time
    print("making train set")
    start = time.time()
    set_imgs = generate_set_images(env, renderer, **generate_set_kwargs)
    set_imgs = ptu.from_numpy(np.array(list(set_imgs)))
    print("making eval set", time.time() - start)
    start = time.time()
    eval_set_imgs = generate_set_images(env, renderer, **generate_set_kwargs)
    eval_set_imgs = ptu.from_numpy(np.array(list(eval_set_imgs)))
    set_imgs_iterator = set_imgs  # an array is already a valid data iterator
    print("making ungrouped images", time.time() - start)
    start = time.time()
    ungrouped_imgs = generate_images(
        env, renderer, num_images=num_ungrouped_images)
    ungrouped_imgs = ptu.from_numpy(np.array(list(ungrouped_imgs)))
    print("done", time.time() - start)
    return eval_set_imgs, renderer, set_imgs, set_imgs_iterator, ungrouped_imgs
