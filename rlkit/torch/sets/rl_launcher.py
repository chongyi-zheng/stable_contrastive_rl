import os.path as osp
import typing
from collections import OrderedDict
from functools import partial

import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.distribution import DictDistribution
from rlkit.data_management.contextual_replay_buffer import (
    ContextualRelabelingReplayBuffer,
)
from rlkit.envs.contextual import (
    ContextualEnv,
    delete_info,
    ContextualRewardFn,
)
from rlkit.envs.contextual.set_distributions import (
    LatentGoalDictDistributionFromSet,
    SetDiagnostics,
    OracleRIGMeanSetter,
    SetReward,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.images import EnvRenderer, InsertImageEnv
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.contextual_path_collector import (
    ContextualPathCollector,
)
from rlkit.samplers.rollout_functions import contextual_rollout
from rlkit.torch.distributions import MultivariateDiagonalNormal
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.sets import set_vae_trainer
from rlkit.torch.sets.models import create_dummy_image_vae
from rlkit.torch.sets.set_creation import create_sets
from rlkit.torch.sets.set_projection import Set
from rlkit.torch.sets.vae_launcher import train_set_vae
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.visualization import video


class NormalLikelihoodRewardFn(ContextualRewardFn):
    def __init__(
            self,
            observation_key,
            mean_key,
            covariance_key,
            batched=True,
            drop_log_det_term=False,
            use_proper_scale_diag=True,
            sqrt_reward=False,
    ):
        self.observation_key = observation_key
        self.mean_key = mean_key
        self.covariance_key = covariance_key
        self.batched = batched
        self.drop_log_det_term = drop_log_det_term
        self.use_proper_scale_diag = use_proper_scale_diag
        self.sqrt_reward = sqrt_reward

    def __call__(self, states, actions, next_states, contexts):
        x = next_states[self.observation_key]
        mean = contexts[self.mean_key]
        covariance = contexts[self.covariance_key]
        if self.drop_log_det_term:
            reward = -((x - mean) ** 2) / (2 * covariance)
            reward = reward.sum(axis=-1)
            if self.sqrt_reward:
                reward = -np.sqrt(-reward)
        else:
            if self.use_proper_scale_diag:
                scale_diag = covariance ** 0.5
            else:
                scale_diag = covariance
            distribution = MultivariateDiagonalNormal(
                loc=ptu.from_numpy(mean), scale_diag=ptu.from_numpy(scale_diag)
            )
            reward = ptu.get_numpy(distribution.log_prob(ptu.from_numpy(x)))
        if not self.batched:
            reward = reward[0]
        return reward


class StateImageGoalDiagnosticsFn:
    def __init__(self, state_to_goal_keys_map):
        self.state_to_goal_keys_map = state_to_goal_keys_map

    def __call__(self, paths, contexts):
        diagnostics = OrderedDict()
        for state_key in self.state_to_goal_keys_map:
            goal_key = self.state_to_goal_keys_map[state_key]
            values = []
            for i in range(len(paths)):
                state = paths[i]["observations"][-1][state_key]
                goal = contexts[i][goal_key]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
            diagnostics_key = goal_key + "/final/distance"
            diagnostics.update(
                create_stats_ordered_dict(diagnostics_key, values,)
            )
        return diagnostics


class FilterKeys(DictDistribution):
    def __init__(self, distribution: DictDistribution, keys_to_keep):
        self.keys_to_keep = keys_to_keep
        self.distribution = distribution
        self._spaces = {
            k: v for k, v in distribution.spaces.items() if k in keys_to_keep
        }

    def sample(self, batch_size: int):
        batch = self.distribution.sample(batch_size)
        return {k: v for k, v in batch.items() if k in self.keys_to_keep}

    @property
    def spaces(self):
        return self._spaces


class OracleMeanSettingPathCollector(ContextualPathCollector):
    def __init__(
            self,
            env: ContextualEnv,
            policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            context_keys_for_policy='context',
            render=False,
            render_kwargs=None,
            **kwargs
    ):
        rollout_fn = partial(
            contextual_rollout,
            context_keys_for_policy=context_keys_for_policy,
            observation_key=observation_key,
        )
        super().__init__(
            env, policy, max_num_epoch_paths_saved, render, render_kwargs,
            rollout_fn=rollout_fn,
            **kwargs
        )
        self._observation_key = observation_key
        self._context_keys_for_policy = context_keys_for_policy

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            context_keys_for_policy=self._context_keys_for_policy,
        )
        return snapshot


class InitStateConditionedContextualEnv(ContextualEnv):
    def reset(self):
        obs = self.env.reset()
        self._rollout_context_batch = self.context_distribution.sample(
            1, init_obs=obs
        )
        self._update_obs(obs)
        self._last_obs = obs
        return obs


def disco_experiment(
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
        exploration_policy_kwargs=None,
        # Video parameters
        save_video=True,
        save_video_kwargs=None,
        renderer_kwargs=None,
):
    if rig_goal_setter_kwargs is None:
        rig_goal_setter_kwargs = {}
    if reward_fn_kwargs is None:
        reward_fn_kwargs = {}
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
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
    expl_env, expl_context_distrib, expl_reward = (
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
        )
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
    context_keys = [
        expl_context_distrib.mean_key,
        expl_context_distrib.covariance_key,
        expl_context_distrib.set_index_key,
        expl_context_distrib.set_embedding_key,
    ]
    if rig:
        context_keys_for_rl = [
            expl_context_distrib.mean_key,
        ]
    else:
        if use_onehot_set_embedding:
            context_keys_for_rl = [
                expl_context_distrib.set_embedding_key,
            ]
        else:
            context_keys_for_rl = [
                expl_context_distrib.mean_key,
                expl_context_distrib.covariance_key,
            ]

    obs_dim = np.prod(expl_env.observation_space.spaces[observation_key].shape)
    obs_dim += sum(
        [np.prod(expl_env.observation_space.spaces[k].shape)
         for k in context_keys_for_rl]
    )
    action_dim = np.prod(expl_env.action_space.shape)

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

    replay_buffer = ContextualRelabelingReplayBuffer(
        env=eval_env,
        context_keys=context_keys,
        observation_keys=list({
            observation_key,
            state_observation_key,
            latent_observation_key
        }),
        observation_key=observation_key,
        context_distribution=FilterKeys(expl_context_distrib, context_keys,),
        sample_context_from_obs_dict_fn=None,
        # RemapKeyFn({context_key: observation_key}),
        reward_fn=eval_reward,
        post_process_batch_fn=concat_context_to_obs,
        **replay_buffer_kwargs,
    )
    trainer = SACTrainer(
        env=expl_env,
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
    exploration_policy = create_exploration_policy(
        expl_env, policy, **exploration_policy_kwargs
    )
    expl_path_collector = ContextualPathCollector(
        expl_env,
        exploration_policy,
        observation_key=observation_key,
        context_keys_for_policy=context_keys_for_rl,
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs,
    )
    algorithm.to(ptu.device)

    if save_video:
        set_index_key = eval_context_distrib.set_index_key
        expl_video_func = DisCoVideoSaveFunction(
            model,
            sets,
            expl_path_collector,
            tag="train",
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
        algorithm.post_train_funcs.append(expl_video_func)

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


def contextual_env_distrib_and_reward(
        vae,
        sets: typing.List[Set],
        state_env,
        renderer,
        reward_fn_kwargs,
        use_ground_truth_reward,
        state_observation_key,
        latent_observation_key,
        example_image_key,
        set_description_key,
        observation_key,
        image_observation_key,
        rig_goal_setter_kwargs,
        oracle_rig_goal=False,
):
    img_env = InsertImageEnv(state_env, renderer=renderer)
    encoded_env = EncoderWrappedEnv(
        img_env,
        vae,
        step_keys_map={image_observation_key: latent_observation_key},
    )
    if oracle_rig_goal:
        context_env_class = InitStateConditionedContextualEnv
        goal_distribution_params_distribution = (
            OracleRIGMeanSetter(
                sets, vae, example_image_key,
                env=state_env,
                renderer=renderer,
                cycle_for_batch_size_1=True,
                **rig_goal_setter_kwargs
            )
        )
    else:
        context_env_class = ContextualEnv
        goal_distribution_params_distribution = (
            LatentGoalDictDistributionFromSet(
                sets, vae, example_image_key, cycle_for_batch_size_1=True,
            )
        )
    if use_ground_truth_reward:
        reward_fn, unbatched_reward_fn = create_ground_truth_set_rewards_fns(
            sets,
            goal_distribution_params_distribution.set_index_key,
            state_observation_key,
        )
    else:
        reward_fn, unbatched_reward_fn = create_normal_likelihood_reward_fns(
            latent_observation_key,
            goal_distribution_params_distribution.mean_key,
            goal_distribution_params_distribution.covariance_key,
            reward_fn_kwargs,
        )
    set_diagnostics = SetDiagnostics(
        set_description_key=set_description_key,
        set_index_key=goal_distribution_params_distribution.set_index_key,
        observation_key=state_observation_key,
    )
    env = context_env_class(
        encoded_env,
        context_distribution=goal_distribution_params_distribution,
        reward_fn=reward_fn,
        unbatched_reward_fn=unbatched_reward_fn,
        observation_key=observation_key,
        contextual_diagnostics_fns=[
            # goal_diagnostics,
            set_diagnostics,
        ],
        update_env_info_fn=delete_info,
    )
    return env, goal_distribution_params_distribution, reward_fn


def create_ground_truth_set_rewards_fns(
        sets,
        set_index_key,
        state_observation_key,
):
    reward_fn = SetReward(
        sets=sets,
        set_index_key=set_index_key,
        observation_key=state_observation_key,
    )
    unbatched_reward_fn = SetReward(
        sets=sets,
        set_index_key=set_index_key,
        observation_key=state_observation_key,
        batched=False,
    )
    return reward_fn, unbatched_reward_fn


def create_normal_likelihood_reward_fns(
        latent_observation_key,
        mean_key,
        covariance_key,
        reward_fn_kwargs,
):
    assert mean_key != covariance_key, "probably a typo"
    reward_fn = NormalLikelihoodRewardFn(
        observation_key=latent_observation_key,
        mean_key=mean_key,
        covariance_key=covariance_key,
        **reward_fn_kwargs
    )
    unbatched_reward_fn = NormalLikelihoodRewardFn(
        observation_key=latent_observation_key,
        mean_key=mean_key,
        covariance_key=covariance_key,
        batched=False,
        **reward_fn_kwargs
    )
    return reward_fn, unbatched_reward_fn


class DisCoVideoSaveFunction:
    def __init__(
        self,
        model,
        sets,
        data_collector,
        tag,
        save_video_period,
        reconstruction_key=None,
        decode_set_image_key=None,
        set_visualization_key=None,
        example_image_key=None,
        set_index_key=None,
        **kwargs
    ):
        self.model = model
        self.sets = sets
        self.data_collector = data_collector
        self.tag = tag
        self.decode_set_image_key = decode_set_image_key
        self.set_visualization_key = set_visualization_key
        self.reconstruction_key = reconstruction_key
        self.example_image_key = example_image_key
        self.set_index_key = set_index_key
        self.dump_video_kwargs = kwargs
        self.save_video_period = save_video_period
        self.keys = []
        self.keys.append("image_observation")
        if reconstruction_key:
            self.keys.append(reconstruction_key)
        if set_visualization_key:
            self.keys.append(set_visualization_key)
            self.keys.append('example0')
            self.keys.append('example1')
            self.keys.append('example2')
            self.keys.append('example3')
        if decode_set_image_key:
            self.keys.append(decode_set_image_key)
        self.logdir = logger.get_snapshot_dir()

    def __call__(self, algo, epoch):
        paths = self.data_collector.get_epoch_paths()
        if epoch % self.save_video_period == 0 or epoch == algo.num_epochs:
            filename = "video_{epoch}_{tag}.mp4".format(
                epoch=epoch, tag=self.tag
            )
            filepath = osp.join(self.logdir, filename)
            self.save_video_of_paths(paths, filepath)

    def save_video_of_paths(self, paths, filepath):
        if self.reconstruction_key:
            for i in range(len(paths)):
                self.add_reconstruction_to_path(paths[i])
        if self.set_visualization_key:
            for i in range(len(paths)):
                self.add_set_visualization_to_path(paths[i])
        if self.decode_set_image_key:
            for i in range(len(paths)):
                self.add_decoded_goal_to_path(paths[i])
        video.dump_paths(
            None, filepath, paths, self.keys, **self.dump_video_kwargs,
        )

    def add_set_visualization_to_path(self, path):
        set_idx = path["full_observations"][0][self.set_index_key]
        set = self.sets[set_idx]
        set_visualization = set.example_dict[self.example_image_key].mean(
            axis=0
        )
        example0 = set.example_dict[self.example_image_key][0]
        example1 = set.example_dict[self.example_image_key][1]
        example2 = set.example_dict[self.example_image_key][2]
        example3 = set.example_dict[self.example_image_key][3]
        for i_in_path, d in enumerate(path["full_observations"]):
            d[self.set_visualization_key] = set_visualization
            d['example0'] = example0
            d['example1'] = example1
            d['example2'] = example2
            d['example3'] = example3

    def add_decoded_goal_to_path(self, path):
        set_idx = path["full_observations"][0][self.set_index_key]
        set = self.sets[set_idx]
        sampled_data = set.example_dict[self.example_image_key]
        posteriors = self.model.encoder(ptu.from_numpy(sampled_data))
        learned_prior = set_vae_trainer.compute_prior(posteriors)
        decoded = self.model.decoder(learned_prior.mean)
        decoded_img = ptu.get_numpy(decoded.mean)[0]
        for i_in_path, d in enumerate(path["full_observations"]):
            d[self.decode_set_image_key] = decoded_img

    def add_reconstruction_to_path(self, path):
        for i_in_path, d in enumerate(path["full_observations"]):
            latent = d["latent_observation"]
            decoded_img = self.model.decode_one_np(latent)
            d[self.reconstruction_key] = np.clip(decoded_img, 0, 1)
