import os.path as osp

from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.action_repeat import ActionRepeatPolicy
from rlkit.samplers.data_collector import VAEWrappedEnvPathCollector
from rlkit.samplers.data_collector.path_collector import GoalConditionedPathCollector
from rlkit.visualization.video import VideoSaveFunction
from rlkit.torch.her.her import HERTrainer
from rlkit.envs.reward_mask_wrapper import DiscreteDistribution, RewardMaskWrapper
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy


def td3_experiment(variant):
    import rlkit.samplers.rollout_functions as rf
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.exploration_strategies.base import (
        PolicyWrappedWithExplorationStrategy
    )

    from rlkit.torch.td3.td3 import TD3 as TD3Trainer
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

    from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy
    # preprocess_rl_variant(variant)
    env = get_envs(variant)
    expl_env = env
    eval_env = env
    es = get_exploration_strategy(variant, env)

    if variant.get("use_masks", False):
        mask_wrapper_kwargs = variant.get("mask_wrapper_kwargs", dict())

        expl_mask_distribution_kwargs = variant["expl_mask_distribution_kwargs"]
        expl_mask_distribution = DiscreteDistribution(**expl_mask_distribution_kwargs)
        expl_env = RewardMaskWrapper(env, expl_mask_distribution, **mask_wrapper_kwargs)

        eval_mask_distribution_kwargs = variant["eval_mask_distribution_kwargs"]
        eval_mask_distribution = DiscreteDistribution(**eval_mask_distribution_kwargs)
        eval_env = RewardMaskWrapper(env, eval_mask_distribution, **mask_wrapper_kwargs)
        env = eval_env

    max_path_length = variant['max_path_length']

    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = variant.get('achieved_goal_key', 'latent_achieved_goal')
    # achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )

    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )


    if variant.get("use_subgoal_policy", False):
        from rlkit.policies.timed_policy import SubgoalPolicyWrapper

        subgoal_policy_kwargs = variant.get('subgoal_policy_kwargs', {})

        policy = SubgoalPolicyWrapper(
            wrapped_policy=policy,
            env=env,
            episode_length=max_path_length,
            **subgoal_policy_kwargs
        )
        target_policy = SubgoalPolicyWrapper(
            wrapped_policy=target_policy,
            env=env,
            episode_length=max_path_length,
            **subgoal_policy_kwargs
        )


    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        # use_masks=variant.get("use_masks", False),
        **variant['replay_buffer_kwargs']
    )

    trainer = TD3Trainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_policy=target_policy,
        **variant['td3_trainer_kwargs']
    )
    # if variant.get("use_masks", False):
    #     from rlkit.torch.her.her import MaskedHERTrainer
    #     trainer = MaskedHERTrainer(trainer)
    # else:
    trainer = HERTrainer(trainer)
    if variant.get("do_state_exp", False):
        eval_path_collector = GoalConditionedPathCollector(
            eval_env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            # use_masks=variant.get("use_masks", False),
            # full_mask=True,
        )
        expl_path_collector = GoalConditionedPathCollector(
            expl_env,
            expl_policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            # use_masks=variant.get("use_masks", False),
        )
    else:
        eval_path_collector = VAEWrappedEnvPathCollector(
            env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            goal_sampling_mode=['evaluation_goal_sampling_mode'],
        )
        expl_path_collector = VAEWrappedEnvPathCollector(
            env,
            expl_policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            goal_sampling_mode=['exploration_goal_sampling_mode'],
        )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    vis_variant = variant.get('vis_kwargs', {})
    vis_list = vis_variant.get('vis_list', [])
    if variant.get("save_video", True):
        if variant.get("do_state_exp", False):
            rollout_function = rf.create_rollout_function(
                rf.multitask_rollout,
                max_path_length=max_path_length,
                observation_key=observation_key,
                desired_goal_key=desired_goal_key,
                # use_masks=variant.get("use_masks", False),
                # full_mask=True,
                # vis_list=vis_list,
            )
            video_func = get_video_save_func(
                rollout_function,
                env,
                policy,
                variant,
            )
        else:
            video_func = VideoSaveFunction(
                env,
                variant,
            )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)
    algorithm.train()


def twin_sac_experiment(variant):
    import rlkit.torch.pytorch_util as ptu
    from rlkit.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer
    from rlkit.torch.networks import ConcatMlp
    from rlkit.torch.sac.policies import TanhGaussianPolicy
    from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
    from rlkit.torch.sac.policies import MakeDeterministic
    from rlkit.torch.sac.sac import SACTrainer

    preprocess_rl_variant(variant)
    env = get_envs(variant)
    max_path_length = variant['max_path_length']
    observation_key = variant.get('observation_key', 'latent_observation')
    desired_goal_key = variant.get('desired_goal_key', 'latent_desired_goal')
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
    obs_dim = (
            env.observation_space.spaces[observation_key].low.size
            + env.observation_space.spaces[desired_goal_key].low.size
    )
    action_dim = env.action_space.low.size
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        **variant['policy_kwargs']
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    trainer = SACTrainer(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['twin_sac_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    if variant.get("do_state_exp", False):
        eval_path_collector = GoalConditionedPathCollector(
            env,
            MakeDeterministic(policy),
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        expl_path_collector = GoalConditionedPathCollector(
            env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
    else:
        eval_path_collector = VAEWrappedEnvPathCollector(
            variant['evaluation_goal_sampling_mode'],
            env,
            MakeDeterministic(policy),
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        expl_path_collector = VAEWrappedEnvPathCollector(
            variant['exploration_goal_sampling_mode'],
            env,
            policy,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=env,
        evaluation_env=env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **variant['algo_kwargs']
    )

    if variant.get("save_video", True):
        video_func = VideoSaveFunction(
            env,
            variant,
        )
        algorithm.post_train_funcs.append(video_func)

    algorithm.to(ptu.device)
    if not variant.get("do_state_exp", False):
        env.vae.to(ptu.device)
    algorithm.train()


def get_presampled_goals_path(path=''):
    """
    :param path: if relative, this will rpe
    :param config: One of a few options:
        - string: the path to the
        - tuple of two strings: the first string specifies the 'mode' and the
            second string specifies extra parameters to that mode
        - None: return None
    :return:  Path to the presampled goals, or None.
    """
    if not path:
        return path
    if path[0] == '/':
        return path
    else:
        import multiworld.envs.mujoco as mwmj
        return osp.join(osp.dirname(mwmj.__file__), path)


def get_envs(variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv
    from rlkit.util.io import load_local_or_remote_file
    from rlkit.torch.vae.conditional_conv_vae import CVAE, CDVAE, ACE, CADVAE, DeltaCVAE

    render = variant.get('render', False)
    vae_path = variant.get("vae_path", None)
    reward_params = variant.get("reward_params", dict())
    init_camera = variant.get("init_camera", None)
    do_state_exp = variant.get("do_state_exp", False)
    presample_goals = variant.get('presample_goals', False)
    presample_image_goals_only = variant.get('presample_image_goals_only', False)
    presampled_goals_path = get_presampled_goals_path(
        variant.get('presampled_goals_path', None))
    vae = load_local_or_remote_file(vae_path) if type(vae_path) is str else vae_path
    if 'env_id' in variant:
        import gym
        import multiworld
        multiworld.register_all_envs()
        env = gym.make(variant['env_id'])
    else:
        env = variant["env_class"](**variant['env_kwargs'])
    if not do_state_exp:
        if isinstance(env, ImageEnv):
            image_env = env
        else:
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
            )
        if presample_goals:
            """
            This will fail for online-parallel as presampled_goals will not be
            serialized. Also don't use this for online-vae.
            """
            if presampled_goals_path is None:
                image_env.non_presampled_goal_img_is_garbage = True
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    env=vae_env,
                    env_id=variant.get('env_id', None),
                    **variant['goal_generation_kwargs']
                )
                del vae_env
            else:
                presampled_goals = load_local_or_remote_file(
                    presampled_goals_path
                ).item()
            del image_env
            image_env = ImageEnv(
                env,
                variant.get('imsize'),
                init_camera=init_camera,
                transpose=True,
                normalize=True,
                presampled_goals=presampled_goals,
                **variant.get('image_env_kwargs', {})
            )
            vae_env = VAEWrappedEnv(
                image_env,
                vae,
                imsize=image_env.imsize,
                decode_goals=render,
                render_goals=render,
                render_rollouts=render,
                reward_params=reward_params,
                presampled_goals = presampled_goals,
                **variant.get('vae_wrapped_env_kwargs', {})
            )
            print("Presampling all goals only")
        else:
            if type(vae) is CVAE or type(vae) is CDVAE or type(vae) is ACE or type(vae) is CADVAE or type(vae) is DeltaCVAE:
                vae_env = ConditionalVAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            else:
                vae_env = VAEWrappedEnv(
                    image_env,
                    vae,
                    imsize=image_env.imsize,
                    decode_goals=render,
                    render_goals=render,
                    render_rollouts=render,
                    reward_params=reward_params,
                    **variant.get('vae_wrapped_env_kwargs', {})
                )
            if presample_image_goals_only:
                presampled_goals = variant['generate_goal_dataset_fctn'](
                    image_env=vae_env.wrapped_env,
                    **variant['goal_generation_kwargs']
                )
                image_env.set_presampled_goals(presampled_goals)
                print("Presampling image goals only")
            else:
                print("Not using presampled goals")

        env = vae_env

    return env


def get_exploration_strategy(variant, env):
    from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
    from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
    from rlkit.exploration_strategies.gaussian_and_epsilon import \
        GaussianAndEpsilonStrategy
    from rlkit.exploration_strategies.ou_strategy import OUStrategy
    from rlkit.exploration_strategies.noop import NoopStrategy

    exploration_type = variant['exploration_type']
    # exploration_noise = variant.get('exploration_noise', 0.1)
    es_kwargs = variant.get('es_kwargs', {})
    if exploration_type == 'ou':
        es = OUStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            **es_kwargs
        )
    elif exploration_type == 'gaussian':
        es = GaussianStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            **es_kwargs
        )
    elif exploration_type == 'epsilon':
        es = EpsilonGreedy(
            action_space=env.action_space,
            # prob_random_action=exploration_noise,
            **es_kwargs
        )
    elif exploration_type == 'gaussian_and_epsilon':
        es = GaussianAndEpsilonStrategy(
            action_space=env.action_space,
            # max_sigma=exploration_noise,
            # min_sigma=exploration_noise,  # Constant sigma
            # epsilon=exploration_noise,
            **es_kwargs
        )
    elif exploration_type == 'noop':
        es = NoopStrategy(
            action_space=env.action_space
        )
    else:
        raise Exception("Invalid type: " + exploration_type)
    return es


def preprocess_rl_variant(variant):
    if variant.get("do_state_exp", False):
        variant['observation_key'] = 'state_observation'
        variant['desired_goal_key'] = 'state_desired_goal'
        variant['achieved_goal_key'] = 'state_achieved_goal'


def get_video_save_func(rollout_function, env, policy, variant):
    from multiworld.core.image_env import ImageEnv
    from rlkit.core import logger
    from rlkit.envs.vae_wrappers import temporary_mode
    from rlkit.visualization.video import dump_video
    logdir = logger.get_snapshot_dir()
    save_period = variant.get('save_video_period', 50)
    do_state_exp = variant.get("do_state_exp", False)
    dump_video_kwargs = variant.get("dump_video_kwargs", dict())
    dump_video_kwargs['horizon'] = variant['max_path_length']

    if do_state_exp:
        imsize = variant.get('imsize')
        dump_video_kwargs['imsize'] = imsize
        image_env = ImageEnv(
            env,
            imsize,
            init_camera=variant.get('init_camera', None),
            transpose=True,
            normalize=True,
        )

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                dump_video(image_env, policy, filename, rollout_function,
                           **dump_video_kwargs)
    else:
        image_env = env
        dump_video_kwargs['imsize'] = env.imsize

        def save_video(algo, epoch):
            if epoch % save_period == 0 or epoch == algo.num_epochs:
                filename = osp.join(logdir,
                                    'video_{epoch}_env.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_env',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
                filename = osp.join(logdir,
                                    'video_{epoch}_vae.mp4'.format(epoch=epoch))
                temporary_mode(
                    image_env,
                    mode='video_vae',
                    func=dump_video,
                    args=(image_env, policy, filename, rollout_function),
                    kwargs=dump_video_kwargs
                )
    return save_video


def create_exploration_policy(
        env,
        policy,
        exploration_version='identity',
        repeat_prob=0.,
        prob_random_action=0.,
        exploration_noise=0.,
):
    # TODO: merge with get_exploration_strategy
    if exploration_version == 'identity':
        return policy
    elif exploration_version == 'occasionally_repeat':
        return ActionRepeatPolicy(policy, repeat_prob=repeat_prob)
    elif exploration_version == 'epsilon_greedy':
        return PolicyWrappedWithExplorationStrategy(
            exploration_strategy=EpsilonGreedy(
                action_space=env.action_space,
                prob_random_action=prob_random_action,
            ),
            policy=policy
        )
    elif exploration_version == 'epsilon_greedy_and_occasionally_repeat':
        policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=EpsilonGreedy(
                action_space=env.action_space,
                prob_random_action=prob_random_action,
            ),
            policy=policy
        )
        return ActionRepeatPolicy(policy, repeat_prob=repeat_prob)
    elif exploration_version == 'epsilon_greedy':
        return PolicyWrappedWithExplorationStrategy(
            exploration_strategy=EpsilonGreedy(
                action_space=env.action_space,
                prob_random_action=prob_random_action,
            ),
            policy=policy
        )
    elif exploration_version == 'ou':
        return PolicyWrappedWithExplorationStrategy(
            exploration_strategy=OUStrategy(
                action_space=env.action_space,
                max_sigma=exploration_noise,
                min_sigma=exploration_noise,
            ),
            policy=policy
        )
    else:
        raise ValueError(exploration_version)
