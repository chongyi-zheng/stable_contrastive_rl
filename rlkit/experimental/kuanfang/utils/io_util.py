import os
import glob

import numpy as np
import joblib
import torch
from torch.utils.data import DataLoader

from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch import pytorch_util as ptu  # NOQA
from rlkit.util.io import load_local_or_remote_file
from rlkit.util.io import get_absolute_path

from rlkit.experimental.kuanfang.vae import vae_datasets


def load_datasets(data_dir,
                  encoding_dir=None,
                  dataset_ctor=None,
                  keys=['train', 'test'],
                  vqvae_mode='zq',
                  is_val_format=True,
                  **kwargs
                  ):
    if dataset_ctor is None:
        dataset_ctor = vae_datasets.VaeDataset

    datasets = {}
    for key in keys:
        if is_val_format:
            if key == 'train':
                data_path = os.path.join(data_dir, 'combined_images.npy')
            elif key == 'test':
                data_path = os.path.join(data_dir, 'combined_test_images.npy')
            else:
                raise ValueError
        else:
            data_path = os.path.join(data_dir, '%s_data.npy' % (key))

        if encoding_dir is None:
            encoding_path = None
        else:
            if vqvae_mode == 'zq':
                encoding_path = os.path.join(encoding_dir,
                                             '%s_encoding.npy' % (key))
            elif vqvae_mode == 'zi':
                encoding_path = os.path.join(encoding_dir,
                                             '%s_zi.npy' % (key))
            else:
                raise ValueError

        dataset = dataset_ctor(
            data_path,
            encoding_path,
            is_val_format=is_val_format,
            **kwargs,
        )

        datasets[key] = dataset

    return datasets


def data_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    return train_loader, test_loader


def load_model(root_dir):
    vqvae_path = os.path.join(root_dir, 'vqvae.pt')
    affordance_path = os.path.join(root_dir, 'affordance.pt')
    classifier_path = os.path.join(root_dir, 'classifier.pt')
    discriminator_path = os.path.join(root_dir, 'discriminator.pt')
    trainer_path = os.path.join(root_dir, 'trainer.pt')

    try:
        vqvae = torch.load(vqvae_path).to(ptu.device)
    except Exception:
        vqvae = joblib.load(vqvae_path).to(ptu.device)

    if os.path.exists(affordance_path):
        affordance = torch.load(affordance_path).to(ptu.device)
        try:
            affordance = torch.load(affordance_path).to(ptu.device)
        except Exception:
            affordance = joblib.load(affordance_path).to(ptu.device)
    else:
        affordance = None

    if os.path.exists(classifier_path):
        classifier = torch.load(classifier_path).to(ptu.device)
    else:
        classifier = None

    if os.path.exists(discriminator_path):
        discriminator = torch.load(discriminator_path).to(ptu.device)
    else:
        discriminator = None

    if os.path.exists(trainer_path):
        trainer_dict = load_local_or_remote_file(trainer_path)
        vf = trainer_dict['trainer/vf'].to(ptu.device)
    else:
        vf = None

    return {
        'vqvae': vqvae,
        'affordance': affordance,
        'classifier': classifier,
        'discriminator': discriminator,
        'vf': vf,
    }


def load_rl(path):
    rl_model_dict = load_local_or_remote_file(path)
    qf1 = rl_model_dict['trainer/qf1']
    qf2 = rl_model_dict['trainer/qf2']
    target_qf1 = rl_model_dict['trainer/target_qf1']
    target_qf2 = rl_model_dict['trainer/target_qf2']
    vf = rl_model_dict['trainer/vf']
    policy = rl_model_dict['trainer/policy']
    # if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
    #     policy.std = policy_kwargs['std']
    #     policy.log_std = np.log(policy.std)
    return {
        'qf1': qf1.to(ptu.device),
        'qf2': qf2.to(ptu.device),
        'target_qf1': target_qf1.to(ptu.device),
        'target_qf2': target_qf2.to(ptu.device),
        'vf': vf.to(ptu.device),
        'policy': policy.to(ptu.device),
    }


def load_path(inputs):
    filenames = glob.glob(get_absolute_path(inputs))
    data = []
    for filename in filenames:
        data_i = load_local_or_remote_file(
            filename,
            delete_after_loading=False)
        data.extend(data_i)

    return data


def build_path(path, obs_dict=None, obs_key=None):
    rewards = []
    path_builder = PathBuilder()

    print('loading path, length', len(
        path['observations']), len(path['actions']))
    H = min(len(path['observations']), len(path['actions']))
    print('actions', np.min(path['actions']), np.max(path['actions']))

    for i in range(H):
        if obs_dict:
            ob = path['observations'][i][obs_key]
            next_ob = path['next_observations'][i][obs_key]
        else:
            ob = path['observations'][i]
            next_ob = path['next_observations'][i]
        action = path['actions'][i]
        reward = path['rewards'][i]
        terminal = path['terminals'][i]
        agent_info = path['agent_infos'][i]
        env_info = path['env_infos'][i]

        reward = np.array([reward]).flatten()
        rewards.append(reward)
        terminal = np.array([terminal]).reshape((1, ))
        path_builder.add_all(
            observations=ob,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    path = path_builder.get_all_stacked()

    return path


def preprocess_val_image(data):
    _shape = list(data.shape[:-3])
    data = np.reshape(data, [-1, 3, 48, 48])
    data = np.transpose(data, [0, 1, 3, 2])
    data = np.reshape(data, _shape + [3, 48, 48])
    data = data - 0.5
    return data


def convert_to_val_image(data):
    data = data + 0.5
    _shape = list(data.shape[:-3])
    data = np.reshape(data, [-1, 3, 48, 48])
    data = np.transpose(data, [0, 1, 3, 2])
    data = np.reshape(data, _shape + [-1])
    return data
