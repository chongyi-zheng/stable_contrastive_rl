import os
import glob

from absl import app
from absl import flags

from roboverse.envs.sawyer_rig_affordances_v6 import SawyerRigAffordancesV6

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants
from rlkit.networks.gaussian_policy import GaussianCNNPolicy

from rlkit.envs.drawer_pnp_push_commands import drawer_pnp_push_commands
from rlkit.learning.stable_contrastive_rl import stable_contrastive_rl_experiment
from rlkit.learning.stable_contrastive_rl import process_args
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
    default_variant = dict(
        imsize=48,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianCNNPolicy,
        policy_kwargs=dict(
            hidden_sizes=[1024, 1024, 1024, 1024],
            std=0.15,
            max_log_std=-1,
            min_log_std=-13,
            std_architecture='shared',
            output_activation=None,
            layer_norm=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[1024, 1024, 1024, 1024],
            representation_dim=16,
            repr_norm=False,
            repr_norm_temp=True,
            repr_log_scale=None,
            twin_q=True,
            layer_norm=True,
            img_encoder_arch='cnn',
            init_w=1E-12,
        ),
        network_type='contrastive_cnn',

        trainer_kwargs=dict(
            discount=0.99,
            lr=3E-4,
            gradient_clipping=None,
            soft_target_tau=5E-3,

            # Contrastive RL default hyperparameters
            bc_coef=0.05,
            use_td=True,
            use_td_cpc=False,
            entropy_coefficient=0.0,
            target_entropy=0.0,

            augment_order=['crop'],
            augment_probability=0.5,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            batch_size=2048,
            start_epoch=-300,
            num_epochs=1,

            num_eval_steps_per_epoch=2000,
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=10000,  # set to a large number
        ),
        replay_buffer_kwargs=dict(
            fraction_next_context=0.0,
            fraction_future_context=1.0,
            fraction_distribution_context=0.0,
            max_size=int(1E6),
            neg_from_the_same_traj=False,
        ),
        online_offline_split=True,
        reward_kwargs=dict(
            obs_type='image',
            reward_type='sparse',
            epsilon=2.0,
            terminate_episode=False,
        ),
        online_offline_split_replay_buffer_kwargs=dict(
            offline_replay_buffer_kwargs=dict(
                fraction_next_context=0.0,
                fraction_future_context=1.0,
                fraction_distribution_context=0.0,
                max_size=int(6E5),
                neg_from_the_same_traj=False,
            ),
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.0,
                fraction_future_context=1.0,
                fraction_distribution_context=0.0,
                max_size=0,
                neg_from_the_same_traj=False,
            ),
            sample_online_fraction=0.6
        ),

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

        launcher_config=dict(
            unpack_variant=True,
            region='us-west-1',
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        use_image=True,

        # Load up existing policy/q-network/value network vs train a new one
        pretrained_rl_path=None,

        eval_seeds=14,
        num_demos=18,

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

        # Goals
        'ground_truth_expl_goals': [True],

    }

    return search_space


def process_variant(variant, data_path):
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0
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
    variant['env_kwargs']['reset_interval'] = variant['reset_interval']

    ########################################
    # Image.
    ########################################
    if variant['use_image']:
        for demo_path in variant['path_loader_kwargs']['demo_paths']:
            demo_path['use_latents'] = False

    ########################################
    # Misc.
    ########################################
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
