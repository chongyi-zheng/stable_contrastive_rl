from rlkit.launchers.vae_exp_launcher_util import (
    train_vae,
    train_vae_and_update_variant,
)

from rlkit.launchers.rl_exp_launcher_util import (
    td3_experiment,
    twin_sac_experiment,
)

def rl_experiment(variant):
    experiment_variant_preprocess(variant)
    rl_variant = variant['rl_variant']
    if not variant['rl_variant'].get('do_state_exp', False):
        train_vae_and_update_variant(variant)
    if rl_variant.get('context_based', False):
        print("Using contexts")
        from rlkit.launchers.contextual.state_based_masking import rl_context_experiment
        rl_context_experiment(rl_variant)
    else:
        print("NOT using contexts")
        if 'sac' in rl_variant['algorithm'].lower():
            twin_sac_experiment(rl_variant)
        else:
            td3_experiment(rl_variant)

    # TODO: add online VAE exps, tdm exps, other baseline exps

def vae_experiment(variant):
    experiment_variant_preprocess(variant)
    train_vae(variant["vae_variant"])

def vae_dataset_experiment(variant):
    experiment_variant_preprocess(variant)

    from rlkit.launchers.vae_exp_launcher_util import generate_vae_dataset
    from inspect import signature

    vae_variant = variant['vae_variant']
    generate_vae_dataset_fctn = vae_variant.get('generate_vae_data_fctn', generate_vae_dataset)
    sig = signature(generate_vae_dataset_fctn)
    if len(sig.parameters) > 1:
        generate_vae_dataset_fctn(**vae_variant['generate_vae_dataset_kwargs'])
    else:
        generate_vae_dataset_fctn(vae_variant['generate_vae_dataset_kwargs'])

def experiment_variant_preprocess(variant):
    vae_variant = variant.get('vae_variant', None)
    rl_variant = variant['rl_variant']
    if 'env_id' in variant:
        assert 'env_class' not in variant
        env_id = variant['env_id']
        rl_variant['env_id'] = env_id
        if vae_variant:
            vae_variant['generate_vae_dataset_kwargs']['env_id'] = env_id
    else:
        env_class = variant['env_class']
        env_kwargs = variant['env_kwargs']
        if vae_variant:
            vae_variant['generate_vae_dataset_kwargs']['env_class'] = (
                env_class
            )
            vae_variant['generate_vae_dataset_kwargs']['env_kwargs'] = (
                env_kwargs
            )
        rl_variant['env_class'] = env_class
        rl_variant['env_kwargs'] = env_kwargs
    init_camera = variant.get('init_camera', None)
    imsize = variant.get('imsize', 84)
    if vae_variant:
        vae_variant['generate_vae_dataset_kwargs']['init_camera'] = (
            init_camera
        )
        vae_variant['generate_vae_dataset_kwargs']['imsize'] = imsize
        vae_variant['imsize'] = imsize
    rl_variant['imsize'] = imsize
    rl_variant['init_camera'] = init_camera