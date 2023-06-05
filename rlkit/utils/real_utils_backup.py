import numpy as np
from copy import deepcopy
from PIL import Image
from torchvision.transforms import ColorJitter, RandomResizedCrop
import torchvision.transforms.functional as F

import os

import pickle


# NOTE TO SELF, RUN ONCE WITH ALL, THEN SPLIT (CHANGE NAME) AND RUN  MULTIPLE
# THIS WILL MAKE IT EASIER TO ITERATE ON THE POLICY

SIZE = 64

crop_prob = 0.95
cond_frac = 0.15  # Use FIRST cond_frac of traj for conditioning
samples_per_traj = 0  # 25 # Number of goals sampled per traj
filter_keyword = ['fixed']  # Only keep these trajectories
repeat_dict = {'pot': 1, 'drawer': 1, 'tray': 1, 'pnp': 1}

jitter = ColorJitter((0.75, 1.25), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))


def filter_files(all_files):
    if len(filter_keyword) == 0:
        return all_files
    filtered_files = []

    for filename in all_files:
        if any([word in filename for word in filter_keyword]):
            filtered_files.append(filename)

    return filtered_files


def repeat_files(all_files):
    new_files = []
    for filename in all_files:
        for key in repeat_dict.keys():
            if key in filename:
                for i in range(repeat_dict[key]):
                    new_files.append(filename)
    return new_files


def aug(x, j, c, do_c):
    if do_c:
        x = F.resized_crop(x, c[0], c[1], c[2], c[3],
                           (SIZE, SIZE), Image.ANTIALIAS)
    else:
        x = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)
    x = np.array(j(x))  # / 255
    img = x.transpose([2, 1, 0]).flatten().astype(np.uint8)
    return img


def filter_keys(dictionary, keep=['latent', 'state', 'epsilon']):
    all_keys = list(dictionary.keys())
    for key in all_keys:
        delete = not any([word in key for word in keep])
        if delete:
            del dictionary[key]


def make_dataset(all_files,  # NOQA
                 save_folder,
                 pretrained_vae_path,
                 output_name,
                 IMAGE_KEY,
                 extra_keys):
    os.makedirs(save_folder, exist_ok=True)

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
            print('COULDNT LOAD ABOVE FILE')
            continue

        data_list = None

        # Check if traj is in specific catagory
        for key in catagorized_data.keys():
            if key in filename:
                data_list = catagorized_data[key]

        # Check not, assign to general
        if data_list is None:
            print(filename)
            data_list = catagorized_data['general']

        for traj_i in range(len(data)):
            # Prepare augmentation
            D = deepcopy(data[traj_i])

            traj = D['observations']
            if len(traj) == 0:
                continue
            try:
                img = traj[0][IMAGE_KEY]
            except KeyError:
                continue

            D['actions'] = np.array(D['actions'])
            D['actions'][:, 3] = np.clip(D['actions'][:, 3], -1, 1)

            img = img[0:270, 90:570, ::-1]
            img = Image.fromarray(img, mode='RGB')

            # Process images
            for t in range(len(traj)):
                if not traj[t]:
                    print(traj_i, t)
                    continue
                img = traj[t][IMAGE_KEY]
                img = img[0:270, 90:570, ::-1]
                img = Image.fromarray(img, mode='RGB')
                img = np.array(img)
                img = img.transpose([2, 1, 0]).flatten().astype(np.uint8)
                traj[t][IMAGE_KEY] = img

            # Encode images
            num_images = len(traj)
            images = np.stack([traj[i][IMAGE_KEY]
                              for i in range(num_images)])
            # Update
            data_list.append(images)
            total_size += num_images
            total_samples += samples_per_traj
            total_traj += 1

            print('Trajectories:', total_traj)
            print('Datapoints:', total_size)
            print('Samples:', total_samples)

    # SAVE TRAJECTORIES FOR REINFORCEMENT LEARNING #
    for key in catagorized_data.keys():
        data_list = catagorized_data[key]
        save_filename = save_folder + output_name + '_' + key + '.npy'
        print('saving', save_filename, 'trajectories', len(data_list))
        if len(data_list) > 0:
            np.save(save_filename, data_list)
