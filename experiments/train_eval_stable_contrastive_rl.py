import os
import glob

from absl import app
from absl import flags

from roboverse.envs.sawyer_rig_affordances_v6 import SawyerRigAffordancesV6

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.experimental.chongyiz.networks.gaussian_policy import GaussianCNNPolicy

from rlkit.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from rlkit.experimental.chongyiz.learning.stable_contrastive_rl import stable_contrastive_rl_experiment
from rlkit.experimental.chongyiz.learning.stable_contrastive_rl import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging


flags.DEFINE_string('data_dir', './data', '')
flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_multi_string(
    'arg_binding', None, 'Variant binding to pass through.')

FLAGS = flags.FLAGS


def get_paths(data_dir):
    data_path = 'env6_td_pnp_push_dataset/'
    data_path = os.path.join(data_dir, data_path)
    paths = glob.glob(os.path.join(data_path, '*demos.pkl'))
    demo_paths = [
        dict(path=path,
             obs_dict=True,
             is_demo=True,
             use_latents=True)
        for path in paths]
    # 1054500 transitions in total
    logging.info('Number of demonstration files: %d' % len(demo_paths))
    logging.info('data_path: %s', data_path)

    return data_path, demo_paths


def get_default_variant(demo_paths):
    # vqvae = os.path.join(data_path, 'pretrained')
    # vqvae = os.path.join(data_path, 'pretrained_aug')

    default_variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianCNNPolicy,
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

            # augment_type='default',
            # augment_params={
            #     'RandomResizedCrop': dict(
            #         scale=(0.9, 1.0),
            #         ratio=(0.9, 1.1),
            #     ),
            #     'ColorJitter': dict(
            #         brightness=(0.75, 1.25),
            #         contrast=(0.9, 1.1),
            #         saturation=(0.9, 1.1),
            #         hue=(-0.1, 0.1),
            #     ),
            #     'RandomCrop': dict(
            #         padding=4,
            #         padding_mode='edge'
            #     ),
            # },
            # augment_order=['RandomResizedCrop', 'ColorJitter'],
            augment_order=['crop'],
            augment_probability=0.95,
            # same_augment_in_a_batch=True,
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

        # # (chongyiz): disable planning for IQL
        # use_expl_planner=False,
        # expl_planner_type='scripted',
        # expl_planner_kwargs=dict(
        #     cost_mode='l2_vf',
        #     buffer_size=0,  # TODO
        #     num_levels=3,
        #     min_dt=15,
        # ),
        # expl_planner_scripted_goals=None,
        # expl_contextual_env_kwargs=dict(
        #     num_planning_steps=16,
        #     fraction_planning=1.0,
        #     subgoal_timeout=30,
        #     # subgoal_reaching_thresh=3.0,
        #     # subgoal_reaching_thresh=-1,
        #     subgoal_reaching_thresh=None,
        #     mode='o',
        # ),
        #
        # # (chongyiz): disable planning for IQL
        # use_eval_planner=False,
        # eval_planner_type='scripted',
        # eval_planner_kwargs=dict(
        #     cost_mode='l2_vf',
        #     buffer_size=0,
        #     num_levels=3,
        #     min_dt=15,
        # ),
        # eval_planner_scripted_goals=None,
        # eval_contextual_env_kwargs=dict(
        #     num_planning_steps=16,
        #     fraction_planning=1.0,
        #     subgoal_timeout=30,
        #     # subgoal_reaching_thresh=3.0,
        #     # subgoal_reaching_thresh=-1,
        #     subgoal_reaching_thresh=None,
        #     mode='o',
        # ),

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

        network_version=0,

        use_image=True,

        # Load up existing policy/q-network/value network vs train a new one
        pretrained_rl_path=None,

        eval_seeds=14,
        num_demos=20,  # (chongyiz): 20 * 74 * 190 = 281200 transitions for env6 dataset

        # Video
        num_video_columns=5,
        save_paths=False,

        # Method Name
        method_name='stable_contrastive_rl',
    )

    return default_variant


def get_search_space():
    ########################################
    # Search Space
    ########################################
    search_space = {
        'env_type': ['td_pnp_push'],

        # Training Parameters
        # Use first 'num_demos' demos for offline data
        'num_demos': [20],

        # Reset environment every 'reset_interval' episodes
        'reset_interval': [1],

        # Training Hyperparameters
        # 'trainer_kwargs.',

        # Goals
        'ground_truth_expl_goals': [True],

    }

    return search_space


def process_variant(variant, data_path):  # NOQA
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0  # NOQA
    if variant['algo_kwargs']['start_epoch'] < 0:
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0  # NOQA
    if variant['pretrained_rl_path'] is not None:
        assert variant['algo_kwargs']['start_epoch'] == 0
    if not variant['use_image']:
        assert variant['trainer_kwargs']['augment_probability'] == 0.0
    env_type = variant['env_type']
    if env_type == 'pnp':
        env_type = 'obj'

    ########################################
    # Set the eval_goals.
    ########################################
    full_open_close_str = ''
    if 'eval_seeds' in variant.keys():
        eval_seed_str = f"_seed{variant['eval_seeds']}"
    else:
        eval_seed_str = ''

    eval_goals = os.path.join(
        data_path,
        'goals_early_stop',
        f'{full_open_close_str}{env_type}_scripted_goals{eval_seed_str}.pkl')
    print('eval_goals: ', eval_goals)

    ########################################
    # Goal sampling modes.
    ########################################
    variant['presampled_goal_kwargs']['eval_goals'] = eval_goals
    variant['path_loader_kwargs']['demo_paths'] = (
        variant['path_loader_kwargs']['demo_paths'][:variant['num_demos']])
    variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))
    variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(float(
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size']))
    variant['replay_buffer_kwargs']['max_size'] = int(float(variant['replay_buffer_kwargs']['max_size']))

    if variant['ground_truth_expl_goals']:
        variant['exploration_goal_sampling_mode'] = 'presampled_images'
        variant['training_goal_sampling_mode'] = 'presampled_images'
        variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        variant['presampled_goal_kwargs']['training_goals'] = eval_goals

    # if variant['only_not_done_goals']:
    #     _old_mode = 'presampled_images'
    #     _new_mode = 'not_done_presampled_images'
    #
    #     if variant['training_goal_sampling_mode'] == _old_mode:
    #         variant['training_goal_sampling_mode'] = _new_mode
    #     if variant['exploration_goal_sampling_mode'] == _old_mode:
    #         variant['exploration_goal_sampling_mode'] = _new_mode
    #     if variant['evaluation_goal_sampling_mode'] == _old_mode:
    #         variant['evaluation_goal_sampling_mode'] = _new_mode

    ########################################
    # Environments.
    ########################################
    variant['env_class'] = SawyerRigAffordancesV6
    variant['env_kwargs']['downsample'] = True
    variant['env_kwargs']['env_obs_img_dim'] = 196
    variant['env_kwargs']['test_env_command'] = (
        drawer_pnp_push_commands[variant['eval_seeds']])

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

    ########################################
    # Misc.
    ########################################
    if variant['reward_kwargs']['reward_type'] in ['sparse', 'onion', 'highlevel']:
        variant['trainer_kwargs']['max_value'] = 0.0
        variant['trainer_kwargs']['min_value'] = -1. / (
            1. - variant['trainer_kwargs']['discount'])

    if 'std' in variant['policy_kwargs']:
        if variant['policy_kwargs']['std'] <= 0:
            variant['policy_kwargs']['std'] = None


def main(_):
    data_path, demo_paths = get_paths(data_dir=FLAGS.data_dir)
    default_variant = get_default_variant(demo_paths)
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
        process_variant(variant, data_path)
        variants.append(variant)

    run_variants(stable_contrastive_rl_experiment,
                 variants,
                 run_id=0,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
