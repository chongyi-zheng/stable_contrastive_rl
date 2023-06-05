import numpy as np
from copy import deepcopy
from PIL import Image
from torchvision.transforms import ColorJitter, RandomResizedCrop
import torchvision.transforms.functional as F

import os

import pickle


# NOTE TO SELF, RUN ONCE WITH ALL, THEN SPLIT (CHANGE NAME) AND RUN  MULTIPLE
# THIS WILL MAKE IT EASIER TO ITERATE ON THE POLICY

SIZE = 48  # TODO
crop_prob = 0.95
jitter = ColorJitter((0.75, 1.25), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))


def augment(x, j, c, do_c, do_j):
    x = F.resize(x, (270, 270))
    if do_c:
        x = F.resized_crop(x, c[0], c[1], c[2], c[3],
                           (SIZE, SIZE), Image.ANTIALIAS)
    else:
        x = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)

    if do_j:
        x = j(x)

    x = np.array(x)
    img = x.transpose([2, 1, 0]).flatten().astype(np.uint8)
    return img


def filter_keys(dictionary,
                keep=['image_observation', 'latent', 'state', 'epsilon']):
    all_keys = list(dictionary.keys())
    for key in all_keys:
        delete = not any([word in key for word in keep])
        if delete:
            del dictionary[key]


def make_image_dataset(all_files,  # NOQA
                       output_dir,
                       output_name,
                       image_key,
                       extra_keys,
                       num_augmentations,
                       ):
    total_size = 0
    total_traj = 0
    total_samples = 0

    catagorized_data = {'general': []}  # {'finetune': []}
    for key in extra_keys:
        catagorized_data[key] = []

    for filename in all_files:
        print(filename)
        try:
            if filename.endswith('.npy'):
                data = np.load(filename, allow_pickle=True)
            elif filename.endswith('.p'):
                data = pickle.load(open(filename, 'rb'))
            else:
                ValueError
        except Exception:
            print('Could not load above file.')
            continue

        data_list = None

        # Check if obs is in specific catagory
        for key in catagorized_data.keys():
            if key in filename:
                data_list = catagorized_data[key]

        # Check not, assign to general
        if data_list is None:
            print(filename)
            data_list = catagorized_data['general']

        for traj_ind in range(len(data)):
            for k in range(num_augmentations):

                # Prepare augmentation
                data_i = deepcopy(data[traj_ind])

                obs = data_i['observations']
                if len(obs) == 0:
                    continue
                try:
                    img = obs[0][image_key]
                except KeyError:
                    continue

                data_i['actions'] = np.array(data_i['actions'])
                data_i['actions'][:, 3] = np.clip(
                    data_i['actions'][:, 3], -1, 1)

                img = img[0:270, 90:570, ::-1]
                img = Image.fromarray(img, mode='RGB')
                x = F.resize(x, (270, 270))
                c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
                j = jitter.get_params(
                    (0.75, 1.25), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1))
                do_c = np.random.uniform() < crop_prob
                do_j = (k != 0)

                # Process images
                for t in range(len(obs)):
                    if not obs[t]:
                        print(traj_ind, t)
                        continue
                    img = obs[t][image_key]
                    img = img[0:270, 90:570, ::-1]
                    img = Image.fromarray(img, mode='RGB')
                    y = augment(img, j, c, do_c, do_j)
                    obs[t][image_key] = y

                # Encode images
                num_images = len(obs)
                images = np.stack([obs[i][image_key]
                                  for i in range(num_images)])
                # Update
                data_list.append(images)
                total_size += num_images
                total_traj += 1

            print('Trajectories:', total_traj)
            print('Datapoints:', total_size)
            print('Samples:', total_samples)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        for key in catagorized_data.keys():
            data_list = catagorized_data[key]
            save_filename = output_dir + output_name + '_' + key + '.npy'
            print('saving', save_filename, 'trajectories', len(data_list))
            if len(data_list) > 0:
                np.save(save_filename, data_list)

    return data_list


def make_traj_dataset(  # NOQA
        all_files,
        image_key,
        extra_keys,
        num_augmentations,
        model=None,
        filter_null_actions=False,
        # pretrained_vae_path,
):
    total_size = 0
    total_traj = 0
    total_samples = 0

    # model = load_local_or_remote_file(pretrained_vae_path)
    # ptu.set_gpu_mode(True)
    # model.to(ptu.device)

    # catagorized_data = {'general': []}  # {'finetune': []}
    # for key in extra_keys:
    #     catagorized_data[key] = []

    data_list = []

    for filename in all_files:
        print(filename)
        try:
            if filename.endswith('.npy'):
                data = np.load(filename, allow_pickle=True)
            elif filename.endswith('.p'):
                data = pickle.load(open(filename, 'rb'))
            else:
                ValueError
        except Exception:
            print('Could not load above file')
            continue

        # data_list = None

        # # Check if obs is in specific catagory
        # for key in catagorized_data.keys():
        #     if key in filename:
        #         data_list = catagorized_data[key]

        # # Check not, assign to general
        # if data_list is None:
        #     print(filename)
        #     data_list = catagorized_data['general']

        for traj_ind in range(len(data)):
            num_valid_steps = 0
            for k in range(num_augmentations):
                # Prepare augmentation
                data_i = deepcopy(data[traj_ind])

                obs = data_i['observations']
                next_obs = data_i['next_observations']
                assert len(obs) == len(next_obs)

                if len(obs) == 0:
                    continue
                try:
                    img = obs[0][image_key]
                except KeyError:
                    continue

                data_i['actions'] = np.array(data_i['actions'])
                data_i['actions'][:, 3] = np.clip(
                    data_i['actions'][:, 3], -1, 1)

                if filter_null_actions:
                    actions = data_i['actions']
                    for t in range(actions.shape[0]):
                        for dim in [0, 1, 2, 4]:
                            if np.abs(actions[t, dim]) <= 0.5:
                                actions[t, dim] = 0.0
                            else:
                                if actions[t, dim] > 0:
                                    actions[t, dim] = 1.0
                                else:
                                    actions[t, dim] = -1.0

                    data_i['actions'] = actions

                img = img[0:270, 90:570, ::-1]
                img = Image.fromarray(img, mode='RGB')
                img = F.resize(img, (270, 270))
                c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
                j = jitter.get_params(
                    (0.75, 1.25), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1))
                do_c = np.random.uniform() < crop_prob
                do_j = (k != 0)

                # Process images
                for t in range(len(obs)):
                    if not obs[t]:
                        print(traj_ind, t)
                        continue
                    img = obs[t][image_key]
                    # print('////')
                    # print(obs[t]['hires_image_observation'].shape)
                    # print(obs[t]['image_observation'].shape)
                    # print(obs[t]['hires_image_observation'].mean())
                    # print(obs[t]['image_observation'].mean())
                    # input()
                    img = img[0:270, 90:570, ::-1]
                    img = Image.fromarray(img, mode='RGB')
                    img = augment(img, j, c, do_c, do_j)
                    obs[t][image_key] = img

                    filter_keys(obs[t])  # Delete unnecesary keys

                    # TODO(kuanfang)
                    if 'hires_image_observation' in obs[t]:
                        del obs[t]['hires_image_observation']

                data_i['observations'] = obs

                # Update
                num_images = len(obs)

                if filter_null_actions:
                    valid_t_list = []
                    for t in range(num_images):
                        valid = False
                        for dim in [0, 1, 2, 4]:
                            if np.abs(data_i['actions'][t, dim]) > 0.1:
                                valid = True

                        if valid:
                            valid_t_list.append(t)

                    for key, value in data_i.items():
                        data_i[key] = [value[tau] for tau in valid_t_list]

                    num_valid_steps = len(valid_t_list)

                data_list.append(data_i)

                total_size += num_images
                total_traj += 1

            print('Trajectories:', total_traj)
            print('Datapoints:', total_size)
            print('Valid Datapoints:', num_valid_steps)
            print('Samples:', total_samples)
            print('len(data_list): ', len(data_list))

        yield data_list
        data_list = []

    if len(data_list) > 0:
        yield data_list


def filter_step_fn(ob, action, next_ob, min_action_value=0.3):
    for dim in [0, 1, 2, 4]:
        if np.abs(action[dim]) > min_action_value:
            return False
    return True
