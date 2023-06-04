import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.contextual_replay_buffer import \
    ContextualRelabelingReplayBuffer
from rlkit.envs.images import EnvRenderer
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.util import io
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector,
)
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sets.models import create_dummy_image_vae
from rlkit.torch.sets.rl_launcher import (
    DisCoVideoSaveFunction,
    contextual_env_distrib_and_reward,
)
from rlkit.torch.sets.set_creation import create_sets
from rlkit.torch.sets.vae_launcher import train_set_vae
from rlkit.torch.torch_rl_algorithm import (
    TorchOfflineBatchRLAlgorithm,
)
from rlkit.envs.contextual.set_distributions import (
    LatentGoalDictDistributionFromSet,
)
from rlkit.util import np_util


def convert_raw_trajectories_into_paths(
        raw_trajectories, vae, sets, reward_fn,
        example_image_key,
):
    latent_goal_distribution = LatentGoalDictDistributionFromSet(
        sets, vae, example_image_key, cycle_for_batch_size_1=True,
    )
    paths = []
    for set_i, _ in enumerate(sets):
        for traj in raw_trajectories:
            paths.append(create_path(
                traj, set_i, vae, reward_fn,
                latent_goal_distribution,
            ))
    return paths


def create_path(trajectory, set_i, vae, reward_fn, latent_goal_distribution):
    traj_len = len(trajectory['actions'])

    def create_obs_dict(state_obs, img_obs, contexts):
        latent_obs = vae.encode_np(img_obs)
        return {
            'state_observation': state_obs,
            'latent_observation': latent_obs,
            'image_observation': img_obs,
            **contexts
        }

    lgd = latent_goal_distribution
    set_i_repeated = np.repeat(np.array([set_i]), traj_len)
    contexts = {
        lgd.mean_key: lgd.means[set_i_repeated],
        lgd.covariance_key: lgd.covariances[set_i_repeated],
        lgd.set_index_key: set_i_repeated,
        lgd.set_embedding_key: np_util.onehot(set_i_repeated, len(lgd.sets)),
    }
    observations = create_obs_dict(
        trajectory['state_observation'],
        trajectory['image_observation'],
        contexts,
    )
    next_observations = create_obs_dict(
        trajectory['next_state_observation'],
        trajectory['next_image_observation'],
        contexts,
    )

    actions = trajectory['actions']
    rewards = reward_fn(observations, actions, next_observations, contexts)
    rewards = rewards.reshape(-1, 1)

    path = dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        terminals=trajectory['terminals'],
        rewards=rewards,
    )
    return path


def count_num_samples(paths):
    return sum([len(path['actions']) for path in paths])


def generate_trajectories(
        snapshot_path,
        max_path_length,
        num_steps,
        save_observation_keys,
):
    ptu.set_gpu_mode(True)
    snapshot = io.load_local_or_remote_file(
        snapshot_path,
        file_type='torch',
    )
    policy = snapshot['exploration/policy']
    env = snapshot['exploration/env']
    observation_key = snapshot['exploration/observation_key']
    context_keys_for_rl = snapshot['exploration/context_keys_for_policy']
    path_collector = ContextualPathCollector(
        env,
        policy,
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_rl,
    )
    policy.to(ptu.device)
    paths = path_collector.collect_new_paths(
        max_path_length,
        num_steps,
        True,
    )

    trajectories = []
    for path in paths:
        trajectory = dict(
            actions=path['actions'],
            terminals=path['terminals'],
        )
        for key in save_observation_keys:
            trajectory[key] = np.array([
                obs[key] for obs in path['full_observations']
            ])
            trajectory['next_' + key] = np.array([
                obs[key] for obs in path['full_next_observations']
            ])
        trajectories.append(trajectory)
    return trajectories


def offline_disco_experiment(
        max_path_length,
        qf_kwargs,
        sac_trainer_kwargs,
        replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        generate_set_for_rl_kwargs,
        # VAE parameters
        create_vae_kwargs,
        vae_trainer_kwargs,
        vae_algo_kwargs,
        data_loader_kwargs,
        generate_set_for_vae_pretraining_kwargs,
        num_ungrouped_images,
        beta_schedule_kwargs=None,
        # Oracle settings
        use_ground_truth_reward=False,
        use_onehot_set_embedding=False,
        use_dummy_model=False,
        observation_key="latent_observation",
        # RIG comparison
        rig_goal_setter_kwargs=None,
        rig=False,
        # Miscellaneous
        reward_fn_kwargs=None,
        # None-VAE Params
        env_id=None,
        env_class=None,
        env_kwargs=None,
        latent_observation_key="latent_observation",
        state_observation_key="state_observation",
        image_observation_key="image_observation",
        set_description_key="set_description",
        example_state_key="example_state",
        example_image_key="example_image",
        # Exploration
        presampled_trajectories_path=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
):
    if rig_goal_setter_kwargs is None:
        rig_goal_setter_kwargs = {}
    if reward_fn_kwargs is None:
        reward_fn_kwargs = {}
    if not save_video_kwargs:
        save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    renderer = EnvRenderer(**renderer_kwargs)

    sets = create_sets(
        env_id,
        env_class,
        env_kwargs,
        renderer,
        example_state_key=example_state_key,
        example_image_key=example_image_key,
        **generate_set_for_rl_kwargs,
    )
    if use_dummy_model:
        model = create_dummy_image_vae(
            img_chw=renderer.image_chw,
            **create_vae_kwargs)
    else:
        model = train_set_vae(
            create_vae_kwargs,
            vae_trainer_kwargs,
            vae_algo_kwargs,
            data_loader_kwargs,
            generate_set_for_vae_pretraining_kwargs,
            num_ungrouped_images,
            env_id=env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
            beta_schedule_kwargs=beta_schedule_kwargs,
            sets=sets,
            renderer=renderer,
        )
    eval_env, eval_context_distrib, eval_reward = (
        contextual_env_distrib_and_reward(
            vae=model,
            sets=sets,
            state_env=get_gym_env(
                env_id, env_class=env_class, env_kwargs=env_kwargs,
            ),
            renderer=renderer,
            reward_fn_kwargs=reward_fn_kwargs,
            use_ground_truth_reward=use_ground_truth_reward,
            state_observation_key=state_observation_key,
            latent_observation_key=latent_observation_key,
            example_image_key=example_image_key,
            set_description_key=set_description_key,
            observation_key=observation_key,
            image_observation_key=image_observation_key,
            rig_goal_setter_kwargs=rig_goal_setter_kwargs,
            oracle_rig_goal=rig,
        )
    )
    if rig:
        context_keys_for_rl = [
            eval_context_distrib.mean_key,
        ]
    else:
        if use_onehot_set_embedding:
            context_keys_for_rl = [
                eval_context_distrib.set_embedding_key,
            ]
        else:
            context_keys_for_rl = [
                eval_context_distrib.mean_key,
                eval_context_distrib.covariance_key,
            ]

    obs_dim = np.prod(eval_env.observation_space.spaces[observation_key].shape)
    obs_dim += sum(
        [np.prod(eval_env.observation_space.spaces[k].shape)
         for k in context_keys_for_rl]
    )
    action_dim = np.prod(eval_env.action_space.shape)

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim, output_size=1, **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    target_qf1 = create_qf()
    target_qf2 = create_qf()

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim, action_dim=action_dim, **policy_kwargs
    )

    def concat_context_to_obs(batch, *args, **kwargs):
        obs = batch["observations"]
        next_obs = batch["next_observations"]
        contexts = [batch[k] for k in context_keys_for_rl]
        batch["observations"] = np.concatenate((obs, *contexts), axis=1)
        batch["next_observations"] = np.concatenate(
            (next_obs, *contexts), axis=1,
        )
        return batch
    context_keys = [
        eval_context_distrib.mean_key,
        eval_context_distrib.covariance_key,
        eval_context_distrib.set_index_key,
        eval_context_distrib.set_embedding_key,
    ]

    raw_trajectories = io.load_local_or_remote_file(
        presampled_trajectories_path
    )
    paths = convert_raw_trajectories_into_paths(
        raw_trajectories, model, sets, eval_reward,
        example_image_key,
    )
    num_samples = count_num_samples(paths)
    replay_buffer = ContextualRelabelingReplayBuffer(
        max_size=num_samples,
        env=eval_env,
        context_keys=context_keys,
        observation_keys_to_save=list({
            observation_key,
            state_observation_key,
            latent_observation_key
        }),
        observation_key=observation_key,
        context_distribution=eval_context_distrib,
        sample_context_from_obs_dict_fn=None,
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs,
    )
    for path in paths:
        replay_buffer.add_path(path, ob_dicts_already_combined=True)
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs,
    )

    eval_path_collector = ContextualPathCollector(
        eval_env,
        MakeDeterministic(policy),
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_rl,
    )

    algorithm = TorchOfflineBatchRLAlgorithm(
        trainer=trainer,
        evaluation_env=eval_env,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs,
    )
    algorithm.to(ptu.device)

    if save_video:
        set_index_key = eval_context_distrib.set_index_key
        eval_video_func = DisCoVideoSaveFunction(
            model,
            sets,
            eval_path_collector,
            tag="eval",
            reconstruction_key="image_reconstruction",
            decode_set_image_key="decoded_set_prior",
            set_visualization_key="set_visualization",
            example_image_key=example_image_key,
            set_index_key=set_index_key,
            columns=len(sets),
            unnormalize=True,
            imsize=48,
            image_format=renderer.output_image_format,
            **save_video_kwargs,
        )
        algorithm.post_train_funcs.append(eval_video_func)

    algorithm.train()
