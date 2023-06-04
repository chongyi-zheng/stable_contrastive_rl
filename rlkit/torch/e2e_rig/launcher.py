import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch import networks
from rlkit.torch.e2e_rig.networks import (
    Vae2Encoder,
)
from rlkit.torch.e2e_rig.vae_sac import End2EndSACTrainer
from rlkit.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.vae.conv_vae import ConvVAE
from rlkit.torch.vae.vae_trainer import VAETrainer
from torch import nn


def pointmass_fixed_goal_experiment(variant):
    _pointmass_fixed_goal_experiment(**variant)


def _pointmass_fixed_goal_experiment(
        vae_latent_size,
        replay_buffer_size,
        cnn_kwargs,
        vae_kwargs,
        policy_kwargs,
        qf_kwargs,
        e2e_trainer_kwargs,
        sac_trainer_kwargs,
        algorithm_kwargs,
        eval_path_collector_kwargs=None,
        expl_path_collector_kwargs=None,
        **kwargs
):
    if expl_path_collector_kwargs is None:
        expl_path_collector_kwargs = {}
    if eval_path_collector_kwargs is None:
        eval_path_collector_kwargs = {}
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.pygame.point2d import Point2DEnv
    from multiworld.core.flat_goal_env import FlatGoalEnv
    env = Point2DEnv(
        images_are_rgb=True,
        render_onscreen=False,
        show_goal=False,
        ball_radius=2,
        render_size=48,
        fixed_goal=(0, 0),
    )
    env = ImageEnv(env, imsize=env.render_size, transpose=True, normalize=True)
    env = FlatGoalEnv(env)#, append_goal_to_obs=True)
    input_width, input_height = env.image_shape

    action_dim = int(np.prod(env.action_space.shape))
    vae = ConvVAE(
        representation_size=vae_latent_size,
        input_channels=3,
        imsize=input_width,
        decoder_output_activation=nn.Sigmoid(),
        # decoder_distribution='gaussian_identity_variance',
        **vae_kwargs)
    encoder = Vae2Encoder(vae)

    def make_cnn():
        return networks.CNN(
            input_width=input_width,
            input_height=input_height,
            input_channels=3,
            output_conv_channels=True,
            output_size=None,
            **cnn_kwargs
        )

    def make_qf():
        return networks.MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(
                encoder,
                networks.Flatten(),
            ),
            output_size=1,
            input_size=action_dim+vae_latent_size,
            **qf_kwargs
        )
    qf1 = make_qf()
    qf2 = make_qf()
    target_qf1 = make_qf()
    target_qf2 = make_qf()
    action_dim = int(np.prod(env.action_space.shape))
    policy_cnn = make_cnn()
    policy = TanhGaussianPolicyAdapter(
        nn.Sequential(policy_cnn, networks.Flatten()),
        policy_cnn.conv_output_flat_size,
        action_dim,
        **policy_kwargs
    )
    eval_env = expl_env = env

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        **eval_path_collector_kwargs
    )
    replay_buffer = EnvReplayBuffer(
        replay_buffer_size,
        expl_env,
    )
    vae_trainer = VAETrainer(vae)
    sac_trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **sac_trainer_kwargs
    )
    trainer = End2EndSACTrainer(
        sac_trainer=sac_trainer,
        vae_trainer=vae_trainer,
        **e2e_trainer_kwargs,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        **expl_path_collector_kwargs
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **algorithm_kwargs
    )
    algorithm.to(ptu.device)
    algorithm.train()
