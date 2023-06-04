import os
import glob

import torch.nn.init
from absl import app
from absl import flags
# from absl import logging
import functools

from roboverse.envs.sawyer_rig_affordances_v6 import SawyerRigAffordancesV6
# from rlkit.experimental.chongyiz.envs.dmc import DmcEnv

import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader  # NOQA
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy
# from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.policies import GaussianCNNPolicy
# from rlkit.torch.networks.cnn import CNN
# from rlkit.torch.networks.cnn import ConcatCNN
# from rlkit.torch.sac.policies import GaussianTwoChannelCNNPolicy
from rlkit.torch.networks.cnn import TwoChannelCNN  # NOQA
from rlkit.torch.networks.cnn import ConcatTwoChannelCNN  # NOQA

# from rlkit.torch.networks import Clamp

from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands  # NOQA
from rlkit.experimental.chongyiz.learning.contrastive_rl import contrastive_rl_experiment
from rlkit.experimental.chongyiz.learning.contrastive_rl import process_args
from rlkit.experimental.kuanfang.utils import arg_util
from rlkit.experimental.kuanfang.utils.logging import logger as logging
from rlkit.experimental.chongyiz.networks.utils import variance_scaling_init_

from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicy  # NOQA
from rlkit.experimental.kuanfang.networks.encoding_networks import EncodingGaussianPolicyV2  # NOQA
# from rlkit.experimental.chongyiz.networks.encoding_networks import ContrastiveEncodingGaussianPolicy
from rlkit.experimental.chongyiz.networks.gaussian_policy import (
    GaussianCNNPolicy,
    GaussianTwoChannelCNNPolicy,
    GoalConditionedGaussianFixedReprPolicy,
)


flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('dataset', 'env6', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_string('pretrained_rl_path', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_integer('run_id', 0, '')
flags.DEFINE_integer('seed', 0, '')
flags.DEFINE_multi_string(
    'arg_binding', None, 'Variant binding to pass through.')

FLAGS = flags.FLAGS


def get_paths(data_dir, dataset):  # NOQA
    # dataset = 'env6_vary_exclusive'

    # VAL Data
    if dataset is None:  # NOQA
        raise ValueError

    elif dataset == 'env6':
        data_path = 'env6_td_pnp_push/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        demo_paths = [
            dict(path=data_path + 'env6_td_pnp_push_demos_{}.pkl'.format(str(i)),  # NOQA
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for i in range(40)]
        # (chongyiz): 562400 transitions in total

    elif 'env6_evalseed' in dataset:
        # (chongyiz): use minimal demos of a specific evaluation seed to debug
        data_path = 'env6_td_pnp_push/'
        eval_seed_data_path = os.path.join(
            data_path,
            'keyframes_trajectories/{}/'.format(dataset[5:]))  # NOQA
        data_path = os.path.join(data_dir, data_path)
        eval_seed_data_path = os.path.join(data_dir, eval_seed_data_path)
        paths = glob.glob(os.path.join(eval_seed_data_path, 'demos*.pkl'))
        demo_paths = [
            dict(path=path,  # NOQA
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths]
        # (chongyiz): 56240 transitions in total

    elif dataset == 'env6_1m':
        data_path = 'env6_td_pnp_push_1m/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths]
        # (chongyiz): 1054500 transitions in total
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'dmc_walker_walk':
        data_path = 'dmc_walker_walk_random/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*.npz'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_vary':
        data_path = 'env6_td_pnp_push_vary_color_angle/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_vary_exclusive':
        data_path = 'env6_td_pnp_push_vary_color_angle/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_view0*drawer*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_vary_exclude_task':
        data_path = 'env6_td_pnp_push_vary_color_angle/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*drawer*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_vary_exclude_scene':
        data_path = 'env6_td_pnp_push_vary_color_angle/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_mixed':
        data_path = 'env6_td_pnp_push_vary_color_angle_mixed_tasks/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_mixed_exclude_task':
        data_path = 'env6_td_pnp_push_vary_color_angle_mixed_tasks/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*drawer*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_mixed_exclude_scene':
        data_path = 'env6_td_pnp_push_vary_color_angle_mixed_tasks/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*.pkl'))
        exclude_paths += glob.glob(
            os.path.join(data_path, 'scene*_*2_demos.pkl'))
        exclude_paths += glob.glob(
            os.path.join(data_path, 'scene*_*3_demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        # (chongyiz): 421800 transitions in total
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_mixed_exclude_scene_1m':
        data_path = 'env6_td_pnp_push_vary_color_angle_mixed_tasks_1m/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*.pkl'))
        exclude_paths += glob.glob(
            os.path.join(data_path, 'scene*_*2_demos.pkl'))
        exclude_paths += glob.glob(
            os.path.join(data_path, 'scene*_*3_demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        # (chongyiz): total number of demos = 108 with 14060 transitions in each demo,
        #   72 demos include 1012320 transitions in total,
        #   108 demos in include 1518480 transitions in total.
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    elif dataset == 'env6_mixed_target_only':
        data_path = 'env6_td_pnp_push_vary_color_angle_mixed_tasks/'  # NOQA
        data_path = os.path.join(data_dir, data_path)
        paths = glob.glob(
            os.path.join(data_path, 'scene0*demos.pkl'))
        exclude_paths = glob.glob(
            os.path.join(data_path, 'scene0_*drawer*demos.pkl'))
        demo_paths = [
            dict(path=path,
                 obs_dict=True,
                 is_demo=True,
                 use_latents=True)
            for path in paths if path not in exclude_paths]
        logging.info('Number of demonstration files: %d' % len(demo_paths))

    else:
        assert False

    logging.info('data_path: %s', data_path)

    return data_path, demo_paths


def get_default_variant(dataset_name, data_path, demo_paths, pretrained_rl_path):
    vqvae = os.path.join(data_path, 'pretrained')
    # vqvae = os.path.join(data_path, 'pretrained_aug')

    default_variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        # policy_class=GaussianTwoChannelCNNPolicy,
        policy_kwargs=dict(
            # hidden_sizes=[256, 256],
            hidden_sizes=[2048, 2048],
            # hidden_sizes=[1024, 1024],
            # hidden_sizes=[128, 128],
            # hidden_sizes=[64, 64],
            std=0.15,  # TODO
            max_log_std=-1,
            min_log_std=-13,  # use value from JAX codebase 1e-6 instead of exp(-2)?
            std_architecture='shared',
            output_activation=None,
            layer_norm=True,
            init_w=1E-12,
            dropout_prob=0.0,
        ),
        # vf_type='mlp',
        qf_kwargs=dict(
            # hidden_sizes=[1024, 1024],
            # hidden_sizes=[1024, 1024],
            # hidden_sizes=[256, 256],
            hidden_sizes=[2048, 2048],
            representation_dim=16,
            repr_norm=False,
            repr_norm_temp=True,
            repr_log_scale=None,
            twin_q=True,
            layer_norm=True,
            # img_encoder_type='shared',
            img_encoder_arch='cnn',
            init_w=1E-12,
        ),
        vf_kwargs=dict(
            hidden_sizes=[1024, 1024],
            representation_dim=16,
            repr_norm=False,
            repr_norm_temp=True,
            repr_log_scale=None,
            twin_v=True,
            layer_norm=True,
            init_w=1E-12,
        ),
        obs_encoder_kwargs=dict(),
        network_type=None,

        # TODO (chongyiz): update trainer_kwargs for contrastive_rl
        trainer_kwargs=dict(
            discount=0.99,  # TODO (chongyiz): use discount from JAX codebase
            lr=3E-4,
            reward_scale=1,
            gradient_clipping=None,  # (chongyiz): Do we need gradient clipping to prevent NAN? No!

            critic_lr_warmup=False,  # warmup the critic learning rate

            soft_target_tau=5E-3,  # TODO (chongyiz): use soft_target_tau from JAX codebase

            random_goals=0.0,
            bc_coef=0.05,
            bc_augmentation=False,

            adv_weighted_loss=False,
            actor_q_loss=True,
            bc_train_val_split=False,

            kld_weight=1.0,  # TODO

            reward_transform_kwargs=dict(m=1, b=0),
            terminal_transform_kwargs=None,

            beta=0.1,
            quantile=0.9,
            clip_score=100,

            fraction_generated_goals=0.0,

            min_value=None,
            max_value=None,

            end_to_end=False,  # TODO
            affordance_weight=100.,

            use_encoding_reward_online=False,
            encoding_reward_thresh=None,

            # Contrastive RL default hyperparameters
            use_td=False,
            vf_ratio_loss=False,
            use_b_squared_td=False,
            use_vf_w=False,
            self_normalized_vf_w=False,
            multiply_batch_size_scale=True,
            add_mc_to_td=False,
            use_gcbc=False,
            entropy_coefficient=None,
            target_entropy=0.0,

            augment_type='default',
            augment_params={
                'RandomResizedCrop': dict(
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                ),
                'ColorJitter': dict(
                    brightness=(0.75, 1.25),
                    contrast=(0.9, 1.1),
                    saturation=(0.9, 1.1),
                    hue=(-0.1, 0.1),
                ),
                'RandomCrop': dict(
                    padding=4,
                    padding_mode='edge'
                ),
            },
            augment_order=['RandomResizedCrop', 'ColorJitter'],
            rad_augment_order=['crop'],
            augment_probability=0.95,
            same_augment_in_a_batch=True,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            batch_size=1024,  # (chongyiz): use larger batch size for contrastive_rl instead of 256
            start_epoch=-100,  # offline epochs
            # TODO (chongyiz): Do we need so many online epochs?
            # num_epochs=1001,  # online epochs
            num_epochs=301,  # online epochs

            num_eval_steps_per_epoch=2000,  # (chongyiz): 5 episodes
            num_expl_steps_per_train_loop=2000,
            # num_eval_steps_per_epoch=1500,
            # num_expl_steps_per_train_loop=1500,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,  # use large number to skip path collection during offline training
        ),
        # (chongyiz): only use future goals from replay buffer
        replay_buffer_kwargs=dict(
            fraction_next_context=0.0,
            fraction_future_context=1.0,
            # DELETEM (chongyiz)
            # fraction_foresight_context=0.0,
            # fraction_perturbed_context=0.0,

            fraction_distribution_context=0.0,
            max_size=int(1E6),
            neg_from_the_same_traj=False,
        ),
        online_offline_split=True,
        reward_kwargs=dict(
            obs_type='latent',
            reward_type='sparse',
            epsilon=3.0,  # TODO
            terminate_episode=True,  # TODO
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            offline_replay_buffer_kwargs=dict(
                # TODO (chongyiz): we need to implement geometric distributed
                #  weights to sample future goals
                #
                # (chongyiz): only use future goals from replay buffer for contrastive_rl
                # cause we construct random goal inside the algorithm
                fraction_next_context=0.0,
                fraction_future_context=1.0,  # For offline data only.

                # DELETEME (chongyiz)
                # fraction_foresight_context=0.0,
                # fraction_perturbed_context=0.0,  # TODO

                fraction_distribution_context=0.0,
                max_size=int(6E5),
                neg_from_the_same_traj=False,
            ),
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.0,
                fraction_future_context=1.0,

                # DELETEME (chongyiz)
                # fraction_foresight_context=0.0,
                # fraction_perturbed_context=0.0,

                fraction_distribution_context=0.0,
                max_size=int(4E5),
                neg_from_the_same_traj=False,
            ),
            sample_online_fraction=0.6
        ),

        # observation_key='latent_observation',
        # goal_key='latent_desired_goal',

        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
            save_video_period=25,
            pad_color=0,
        ),

        pretrained_vae_path=vqvae,

        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
            demo_train_split=0.95,
            # min_path_length=15,  # TODO
            add_demos_to_replay_buffer=True,  # set to false if we want to save paths.
            demos_saving_path=None,  # need to be a valid path if add_demos_to_replay_buffer is false
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=48,
            height=48,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

        evaluation_goal_sampling_mode='presampled_images',
        exploration_goal_sampling_mode='presampled_images',
        training_goal_sampling_mode='presample_latents',

        presampled_goal_kwargs=dict(
            eval_goals='',  # HERE
            eval_goals_kwargs={},
            expl_goals='',
            expl_goals_kwargs={},
            training_goals='',
            training_goals_kwargs={},
        ),

        # (chongyiz): disable planning for IQL
        use_expl_planner=False,
        expl_planner_type='scripted',
        expl_planner_kwargs=dict(
            cost_mode='l2_vf',
            buffer_size=0,  # TODO
            num_levels=3,
            min_dt=15,
        ),
        expl_planner_scripted_goals=None,
        expl_contextual_env_kwargs=dict(
            num_planning_steps=16,
            fraction_planning=1.0,
            subgoal_timeout=30,
            # subgoal_reaching_thresh=3.0,
            # subgoal_reaching_thresh=-1,
            subgoal_reaching_thresh=None,
            mode='o',
        ),

        # (chongyiz): disable planning for IQL
        use_eval_planner=False,
        eval_planner_type='scripted',
        eval_planner_kwargs=dict(
            cost_mode='l2_vf',
            buffer_size=0,
            num_levels=3,
            min_dt=15,
        ),
        eval_planner_scripted_goals=None,
        eval_contextual_env_kwargs=dict(
            num_planning_steps=16,
            fraction_planning=1.0,
            subgoal_timeout=30,
            # subgoal_reaching_thresh=3.0,
            # subgoal_reaching_thresh=-1,
            subgoal_reaching_thresh=None,
            mode='o',
        ),

        scripted_goals=None,

        expl_reset_interval=0,

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',  # HERE
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        trainer_type='vib',  # TODO (chongyiz): maybe update trainer_type to switch between different algorithms.
        network_version=0,

        use_image=False,
        finetune_with_obs_encoder=False,
        pretrained_rl_path=pretrained_rl_path,
        eval_seeds=14,
        goal_timeoutk=-1,
        num_demos=20,  # (chongyiz): 20 * 74 * 190 = 281200 transitions for env6 dataset

        # VIP and R3M
        vip_gcbc=False,
        r3m_gcbc=False,

        # VIB
        obs_encoding_dim=64,  # TODO
        affordance_encoding_dim=8,

        policy_class_name='v1',
        use_encoder_in_policy=False,
        fix_encoder_online=True,

        # Video
        num_video_columns=5,
        save_paths=False,

        # Method Name
        method_name='contrastive_nce',

        # Dataset Name
        dataset_name=dataset_name,

    )

    return default_variant


def get_search_space():
    ########################################
    # Search Space
    ########################################
    search_space = {
        # If 'use_multiple_goals'=False use this evaluation environment seed
        'env_type': ['td_pnp_push'],

        # Training Parameters

        'trainer_kwargs.bc': [False],  # Run BC experiment
        # Length of trajectory during exploration and evaluation
        # Reset environment every 'reset_interval' episodes
        # 'env_kwargs.reset_interval': [1],  # Kuan: This is not used.
        'reset_interval': [1],

        # Training Hyperparameters

        # Overrides currently beta with beta_online during finetuning
        'trainer_kwargs.use_online_beta': [False],
        'trainer_kwargs.beta_online': [0.01],

        # Anneal beta every 'anneal_beta_every' by 'anneal_beta_by until
        # 'anneal_beta_stop_at'
        'trainer_kwargs.use_anneal_beta': [False],
        'trainer_kwargs.anneal_beta_every': [20],
        'trainer_kwargs.anneal_beta_by': [.05],
        'trainer_kwargs.anneal_beta_stop_at': [.0001],

        # If True, use pretrained reward classifier. If False, use epsilon.
        'reward_kwargs.use_pretrained_reward_classifier_path': [False],

        'trainer_kwargs.use_online_quantile': [False],
        'trainer_kwargs.quantile_online': [0.99],

        # Goals
        'use_both_ground_truth_and_affordance_expl_goals': [False],
        # If 'use_ground_truth_and_affordance_expl_goals'=True, this gives
        # sampling proportion of affordance model during expl
        'affordance_sampling_prob': [1],
        # If ''use_ground_truth_and_affordance_expl_goals'=False, we use either
        # PixelCNN expl goals or ground truth expl goals
        'ground_truth_expl_goals': [True],

        # For ground truth goals only select goals that are not achieved by the
        # initialization
        'only_not_done_goals': [False],
        # Initialize drawer to fully close or fully open. Alternative,
        # initialized uniform random.
        # 'env_kwargs.full_open_close_init_and_goal': [False],
        # # Only use ground truth goals that are near-fully open or closed.
        # 'env_kwargs.always_goal_is_open': [False],
        # # Only use ground truth goals that are near-fully open or closed.
        # 'full_open_close_goal': [False],

        # Relabeling
        'online_offline_split_replay_buffer_kwargs.online_replay_buffer_kwargs.fraction_distribution_context': [0.0],  # NOQA
    }

    return search_space


def process_variant(dataset, variant, data_path):  # NOQA
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0  # NOQA
    if variant['algo_kwargs']['start_epoch'] < 0:
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0  # NOQA
    if variant['pretrained_rl_path'] is not None:
        assert variant['algo_kwargs']['start_epoch'] == 0
    if variant['trainer_kwargs']['use_online_beta']:
        assert variant['trainer_kwargs']['use_anneal_beta'] is False
    if not variant['use_image']:
        assert variant['trainer_kwargs']['augment_probability'] == 0.0
    env_type = variant['env_type']
    if dataset != 'val' and env_type == 'pnp':
        env_type = 'obj'

    ########################################
    # Set the eval_goals.
    ########################################
    # if variant['full_open_close_goal']:
    #     full_open_close_str = 'full_open_close_'
    # else:
    full_open_close_str = ''
    if 'eval_seeds' in variant.keys():
        eval_seed_str = f"_seed{variant['eval_seeds']}"
    else:
        eval_seed_str = ''

    # if variant['planner_type'] == 'scripted':
    #     scripted_goals = os.path.join(data_path, f'{full_open_close_str}{env_type}_scripted_goals{eval_seed_str}.pkl')  # NOQA
    # elif variant['planner_type'] in ['choice', 'rchoice', 'uchoice']:
    #     scripted_goals = os.path.join(data_path, f'{full_open_close_str}{env_type}_random_goals{eval_seed_str}.pkl')  # NOQA
    # else:
    #     scripted_goals = None
    # variant['scripted_goals'] = scripted_goals

    # TODO
    if variant['goal_timeoutk'] is None:
        if variant['expl_planner_type'] == 'scripted':
            eval_goals = os.path.join(data_path, f'{full_open_close_str}{env_type}_scripted_goals{eval_seed_str}.pkl')  # NOQA
        else:
            eval_goals = os.path.join(data_path, f'{full_open_close_str}{env_type}_goals{eval_seed_str}.pkl')  # NOQA
    else:
        timeoutk_str = f"_timeoutk{variant['goal_timeoutk']}"
        eval_goals = os.path.join(
            data_path,
            'goals_early_stop',
            f'{full_open_close_str}{env_type}_scripted_goals{timeoutk_str}{eval_seed_str}.pkl')  # NOQA
        print('eval_goals: ', eval_goals)

    # Lines below are for reset-free learning, which is unused for now.
    # if variant['expl_planner_scripted_goals'] is not None:
    #     variant['expl_planner_scripted_goals'] = os.path.join(data_path, variant['expl_planner_scripted_goals'])  # NOQA
    # if variant['eval_planner_scripted_goals'] is not None:
    #     variant['eval_planner_scripted_goals'] = os.path.join(data_path, variant['eval_planner_scripted_goals'])  # NOQA

    ########################################
    # Goal sampling modes.
    ########################################
    variant['presampled_goal_kwargs']['eval_goals'] = eval_goals
    variant['path_loader_kwargs']['demo_paths'] = (
        variant['path_loader_kwargs']['demo_paths'][:variant['num_demos']])
    # variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = min(  # NOQA
    #     int(6E5), int(500*75*variant['num_demos']))
    # variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = min(  # NOQA
    #     int(4/6 * 500*75*variant['num_demos']),
    #     int(1E6 - variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))  # NOQA
    variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))
    variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size']))
    variant['replay_buffer_kwargs']['max_size'] = int(float(variant['replay_buffer_kwargs']['max_size']))

    if variant['use_both_ground_truth_and_affordance_expl_goals']:
        variant['exploration_goal_sampling_mode'] = (
            'conditional_vae_prior_and_not_done_presampled_images')
        variant['training_goal_sampling_mode'] = 'presample_latents'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['expl_goals_kwargs']['affordance_sampling_prob'] = variant['affordance_sampling_prob']  # NOQA
    elif variant['ground_truth_expl_goals']:
        # 'presample_latents'
        variant['exploration_goal_sampling_mode'] = 'presampled_images'
        variant['training_goal_sampling_mode'] = 'presampled_images'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['training_goals'] = eval_goals

    if variant['only_not_done_goals']:
        # if variant['env_kwargs']['use_multiple_goals']:
        #     # Convert all 'presampled_images' to
        #     # 'multiple_goals_not_done_presampled_images'
        #     _old_mode = 'presampled_images'
        #     _new_mode = 'multiple_goals_not_done_presampled_images'
        # else:
        # Convert all 'presampled_images' to 'not_done_presampled_images'
        _old_mode = 'presampled_images'
        _new_mode = 'not_done_presampled_images'

        if variant['training_goal_sampling_mode'] == _old_mode:
            variant['training_goal_sampling_mode'] = _new_mode
        if variant['exploration_goal_sampling_mode'] == _old_mode:
            variant['exploration_goal_sampling_mode'] = _new_mode
        if variant['evaluation_goal_sampling_mode'] == _old_mode:
            variant['evaluation_goal_sampling_mode'] = _new_mode

    ########################################
    # Environments.
    ########################################
    if dataset is None:
        raise ValueError
    elif dataset == 'val':
        if env_type in ['top_drawer', 'bottom_drawer']:
            variant['env_class'] = SawyerRigAffordancesV0
            variant['env_kwargs']['env_type'] = env_type
        if env_type == 'tray':
            variant['env_class'] = SawyerRigMultiobjTrayV0
        if env_type == 'pnp':
            variant['env_class'] = SawyerRigMultiobjV0
    elif dataset in ['reset-free', 'tray-reset-free', 'tray-test-reset-free']:
        variant['env_class'] = SawyerRigAffordancesV0
        variant['env_kwargs']['env_type'] = env_type
    elif dataset in ['rotated-top-drawer-reset-free',
                     'reconstructed-rotated-top-drawer-reset-free']:
        variant['env_class'] = SawyerRigAffordancesV1
    elif dataset in [
        'antialias-rotated-top-drawer-reset-free',
        'antialias-right-top-drawer-reset-free',
        'antialias-rotated-semicircle-top-drawer-reset-free',
        'new-view-antialias-rotated-semicircle-top-drawer-reset-free',
        'new-view-antialias-rotated-semicircle-top-drawer-reset-free-large',
        'new-close-view-antialias-rotated-semicircle-top-drawer-reset-free',
    ]:
        variant['env_class'] = SawyerRigAffordancesV1
        variant['env_kwargs']['downsample'] = True
        variant['env_kwargs']['env_obs_img_dim'] = 196
        if dataset == 'antialias-right-top-drawer-reset-free':
            variant['env_kwargs']['fix_drawer_orientation'] = True
        elif dataset == 'antialias-rotated-semicircle-top-drawer-reset-free':
            variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
        elif dataset in [
                'new-view-antialias-rotated-semicircle-top-drawer-reset-free',  # NOQA
                'new-view-antialias-rotated-semicircle-top-drawer-reset-free-large']:    # NOQA
            variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
            variant['env_kwargs']['new_view'] = True
        elif dataset == 'new-close-view-antialias-rotated-semicircle-top-drawer-reset-free':  # NOQA
            variant['env_kwargs']['fix_drawer_orientation_semicircle'] = True
            variant['env_kwargs']['new_view'] = True
            variant['env_kwargs']['close_view'] = True
        else:
            assert False
    elif dataset in [
        'td_pnp_push',
    ]:
        variant['env_class'] = SawyerRigAffordancesV3
        variant['env_kwargs']['downsample'] = True
        variant['env_kwargs']['env_obs_img_dim'] = 196
        variant['env_kwargs']['test_env_command'] = (
            drawer_pnp_push_commands[variant['eval_seeds']])
    elif dataset in [
        'env5_td_pnp_push',
        'env5_td_pnp_push_v2',
    ]:
        variant['env_class'] = SawyerRigAffordancesV5
        variant['env_kwargs']['downsample'] = True
        variant['env_kwargs']['env_obs_img_dim'] = 196
        variant['env_kwargs']['test_env_command'] = (
            drawer_pnp_push_commands[variant['eval_seeds']])
    elif dataset in [
        'env6',
        'env6_1m',
        'env6_vary',
        'env6_vary_exclusive',
        'env6_vary_exclude_task',
        'env6_vary_exclude_scene',
        'env6_mixed',
        'env6_mixed_exclude_task',
        'env6_mixed_exclude_scene',
        'env6_mixed_exclude_scene_1m',
        'env6_mixed_target_only',
    ] or 'env6_evalseed' in dataset:
        variant['env_class'] = SawyerRigAffordancesV6
        variant['env_kwargs']['downsample'] = True
        variant['env_kwargs']['env_obs_img_dim'] = 196
        variant['env_kwargs']['test_env_command'] = (
            drawer_pnp_push_commands[variant['eval_seeds']])
    elif dataset in [
        'dmc_walker_walk'
    ]:
        raise NotImplementedError
        variant['env_class'] = DmcEnv
        variant['env_kwargs']['task'] = 'walker_walk'
        variant['env_kwargs']['size'] = (48, 48)
        variant['env_kwargs']['action_repeat'] = 2
        variant['env_kwargs']['use_goal_idx'] = False
        variant['env_kwargs']['log_per_goal'] = True
    else:
        raise ValueError

    # if 'eval_seeds' in variant.keys():
    #     variant['env_kwargs']['test_env_seed'] = (
    #         variant['eval_seeds'])

    # if variant['env_kwargs']['use_multiple_goals']:
    #     variant['env_kwargs']['test_env_seeds'] = (
    #         variant['multiple_goals_eval_seeds'])

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        variant['policy_class'] = GaussianCNNPolicy
        # variant['policy_class'] = GaussianTwoChannelCNNPolicy
        # variant['qf_class'] = ConcatTwoChannelCNN
        # variant['vf_class'] = TwoChannelCNN

        variant['obs_encoder_kwargs'] = dict()

        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

        # (chongyiz): we can stick with large replay buffer for image encoder exps
        # variant['replay_buffer_kwargs']['max_size'] = int(5E5)
        # variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(2E5)  # NOQA
        # variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(3E5)  # NOQA

    if variant['vip_gcbc']:
        assert variant['trainer_kwargs']['bc_coef'] == 1.0
        variant['policy_class'] = GoalConditionedGaussianFixedReprPolicy

    elif variant['r3m_gcbc']:
        assert variant['trainer_kwargs']['bc_coef'] == 1.0
        raise NotImplementedError


    ########################################
    # Misc.
    ########################################
    # TODO(kuanfang)
    if variant['reward_kwargs']['reward_type'] in ['sparse', 'onion', 'highlevel']:  # NOQA
        variant['trainer_kwargs']['max_value'] = 0.0
        variant['trainer_kwargs']['min_value'] = -1. / (
            1. - variant['trainer_kwargs']['discount'])

    # TODO(kuanfang)
    # if dataset in ['env6']:
    #     if variant['expl_planner_type'] == 'mppi':
    #         variant['pretrained_vae_path'] = os.path.join(
    #             variant['pretrained_vae_path'], 'dt15')  # TODO

    if 'std' in variant['policy_kwargs']:
        if variant['policy_kwargs']['std'] <= 0:
            variant['policy_kwargs']['std'] = None

    # if variant['trainer_type'] == 'sac':
    #     assert not variant['image']
    #     variant['policy_class'] = TanhGaussianPolicy
    #     variant['algo_kwargs']['start_epoch'] = 0
    #     variant['policy_kwargs']['std'] = None
    #     del variant['policy_kwargs']['max_log_std']
    #     del variant['policy_kwargs']['min_log_std']
    #     del variant['policy_kwargs']['std_architecture']
    #
    #     variant['trainer_kwargs'] = dict(
    #         discount=0.995,
    #         lr=3E-4,
    #         reward_scale=1,
    #     )
    if variant['use_encoder_in_policy']:
        if variant['policy_class_name'] == 'v1':
            variant['policy_class'] = EncodingGaussianPolicy
        elif variant['policy_class_name'] == 'v2':
            variant['policy_class'] = EncodingGaussianPolicyV2
        else:
            raise ValueError


def main(_):
    data_path, demo_paths = get_paths(data_dir=FLAGS.data_dir,
                                      dataset=FLAGS.dataset)
    default_variant = get_default_variant(
        FLAGS.dataset,
        data_path,
        demo_paths,
        FLAGS.pretrained_rl_path,
    )
    search_space = get_search_space()

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=default_variant,
    )

    logging.info('arg_binding: ')
    logging.info(FLAGS.arg_binding)

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variant = arg_util.update_bindings(variant,
                                           FLAGS.arg_binding,
                                           check_exist=True)
        process_variant(FLAGS.dataset, variant, data_path)
        variants.append(variant)

    run_variants(contrastive_rl_experiment,
                 variants,
                 run_id=FLAGS.run_id,  # TODO
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
