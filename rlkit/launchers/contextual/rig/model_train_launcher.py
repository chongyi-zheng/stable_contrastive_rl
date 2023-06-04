from rlkit.launchers.contextual.rig.dataset_launcher import generate_vae_dataset

def train_vae(
        variant, env_kwargs, env_id, env_class, imsize, init_camera, return_data=False
    ):
        from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
        from rlkit.torch.vae.conv_vae import (
            ConvVAE,
            SpatialAutoEncoder,
            AutoEncoder,
        )
        import rlkit.torch.vae.conv_vae as conv_vae
        from rlkit.torch.vae.vae_trainer import ConvVAETrainer
        from rlkit.core import logger
        import rlkit.torch.pytorch_util as ptu
        from rlkit.pythonplusplus import identity
        import torch

        logger.remove_tabular_output(
            'progress.csv', relative_to_snapshot_dir=True
        )
        logger.add_tabular_output(
            'model_progress.csv', relative_to_snapshot_dir=True
        )

        beta = variant["beta"]
        representation_size = variant.get("representation_size",
            variant.get("latent_sizes", variant.get("embedding_dim", None)))
        use_linear_dynamics = variant.get('use_linear_dynamics', False)
        generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                                generate_vae_dataset)
        variant['generate_vae_dataset_kwargs']['use_linear_dynamics'] = use_linear_dynamics
        variant['generate_vae_dataset_kwargs']['batch_size'] = variant['algo_kwargs']['batch_size']
        train_dataset, test_dataset, info = generate_vae_dataset_fctn(
            env_kwargs, env_id, env_class, imsize, init_camera,
            **variant['generate_vae_dataset_kwargs']
        )

        if use_linear_dynamics:
            action_dim = train_dataset.data['actions'].shape[2]

        logger.save_extra_data(info)
        logger.get_snapshot_dir()
        if 'beta_schedule_kwargs' in variant:
            beta_schedule = PiecewiseLinearSchedule(
                **variant['beta_schedule_kwargs'])
        else:
            beta_schedule = None
        if 'context_schedule' in variant:
            schedule = variant['context_schedule']
            if type(schedule) is dict:
                context_schedule = PiecewiseLinearSchedule(**schedule)
            else:
                context_schedule = ConstantSchedule(schedule)
            variant['algo_kwargs']['context_schedule'] = context_schedule
        if variant.get('decoder_activation', None) == 'sigmoid':
            decoder_activation = torch.nn.Sigmoid()
        else:
            decoder_activation = identity
        architecture = variant['vae_kwargs'].get('architecture', None)
        if not architecture and imsize == 84:
            architecture = conv_vae.imsize84_default_architecture
        elif not architecture and imsize == 48:
            architecture = conv_vae.imsize48_default_architecture
        variant['vae_kwargs']['architecture'] = architecture
        variant['vae_kwargs']['imsize'] = imsize

        if variant['algo_kwargs'].get('is_auto_encoder', False):
            model = AutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        elif variant.get('use_spatial_auto_encoder', False):
            model = SpatialAutoEncoder(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        else:
            vae_class = variant.get('vae_class', ConvVAE)
            if use_linear_dynamics:
                model = vae_class(representation_size, decoder_output_activation=decoder_activation, action_dim=action_dim,**variant['vae_kwargs'])
            else:
                model = vae_class(representation_size, decoder_output_activation=decoder_activation,**variant['vae_kwargs'])
        model.to(ptu.device)

        vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
        trainer = vae_trainer_class(model, beta=beta,
                           beta_schedule=beta_schedule,
                           **variant['algo_kwargs'])
        save_period = variant['save_period']

        dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)
        for epoch in range(variant['num_epochs']):
            should_save_imgs = (epoch % save_period == 0)
            trainer.train_epoch(epoch, train_dataset)
            trainer.test_epoch(epoch, test_dataset)

            if should_save_imgs:
                trainer.dump_reconstructions(epoch)
                trainer.dump_samples(epoch)
                if dump_skew_debug_plots:
                    trainer.dump_best_reconstruction(epoch)
                    trainer.dump_worst_reconstruction(epoch)
                    trainer.dump_sampling_histogram(epoch)

            stats = trainer.get_diagnostics()
            for k, v in stats.items():
                logger.record_tabular(k, v)
            logger.dump_tabular()
            trainer.end_epoch(epoch)

            if epoch % 50 == 0:
                logger.save_itr_params(epoch, model)
        logger.save_extra_data(model, 'vae.pkl', mode='pickle')

        logger.remove_tabular_output(
            'model_progress.csv',
            relative_to_snapshot_dir=True,
        )
        logger.add_tabular_output(
            'progress.csv',
            relative_to_snapshot_dir=True,
        )
        if return_data:
            return model, train_dataset, test_dataset
        return model
