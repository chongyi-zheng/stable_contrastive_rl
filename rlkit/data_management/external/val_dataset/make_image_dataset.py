import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from copy import deepcopy
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
import torchvision.transforms.functional as F
from torchvision.utils import save_image
# import rlkit.torch.pytorch_util as ptu
import random
import sys
import os
import pickle

AUGMENT = 10
SIZE = 48
crop_prob = 0.95

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
cropper = RandomResizedCrop((SIZE, SIZE), (0.9, 1.0), (0.9, 1.1))
cond_frac = 0.15 # Use any of the first cond_frac of traj for conditioning
filter_keyword = ['fixed',] # Only keep these trajectories
repeat_dict = {'pot': 3, 'drawer': 2, 'tray': 1, 'pnp': 1}
save_keyword = 'uniform'

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
    if do_c: x = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    else: x = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)
    x = np.array(j(x))
    img = x.transpose([2, 1, 0]).flatten()
    return img

def split_data(dataset_size, test_p=0.95):
    unaugmented_dataset_size = dataset_size / AUGMENT
    num_train = int(unaugmented_dataset_size * test_p)
    indices = np.arange(unaugmented_dataset_size)
    np.random.shuffle(indices)
    unaugmented_train_indices, unaugmented_test_indices = indices[:num_train], indices[num_train:]
    
    train_indices = []
    for i in unaugmented_train_indices:
        train_indices.extend([int(AUGMENT*i+j) for j in range(4)])

    test_indices = []
    for i in unaugmented_test_indices:
        test_indices.extend([int(AUGMENT*i+j) for j in range(4)])
    
    return train_indices, test_indices

# all_files = glob.glob("/media/ashvin/data1/s3doodad/demos/icra2021/dataset_v4/*") + glob.glob("/media/ashvin/data1/s3doodad/demos/icra2021/dataset_v4a/*") + glob.glob("/media/ashvin/data1/s3doodad/demos/icra2021/dataset_v4b/*")
# all_files = filter_files(all_files)
# all_files = repeat_files(all_files)

def make_dataset(all_files, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    # ptu.set_gpu_mode(True)
    obs, envs, dataset = [], [], {}
    goal_obs, goal_envs, goal_dataset = [], [], {}
    total_size = 0

    print('Predicted Size: ', len(all_files) * 10 * 70 * AUGMENT) # files * traj_per_file * trans_per_traj * augment
    random.shuffle(all_files)

    for filename, load_params in all_files:
        is_demo = load_params['is_demo']
        image_key = load_params['image_key']

        print(filename)
        try:
            if filename.endswith(".npy"):
                data = np.load(filename, allow_pickle=True)
            elif filename.endswith(".p"):
                data = pickle.load(open(filename, "rb"))
            else:
                error
        except:
            print("COULDNT LOAD ABOVE FILE")
            continue
        for traj_i in range(len(data)):
            for _ in range(AUGMENT):
                D = deepcopy(data[traj_i])
                traj = D["observations"]
                if len(traj) == 0:
                    continue
                try: img = traj[0][image_key]
                except KeyError: continue

                num_images = len(traj)
                goal_timestep = int(cond_frac * num_images)

                # Crop Images
                traj_images = np.stack([traj[i][image_key][0:270, 90:570, ::-1].copy() 
                    for i in range(num_images)])

                for t in range(len(traj)):
                    if not traj[t]: continue
                    
                    # Prepare Image
                    img = Image.fromarray(traj_images[t], mode='RGB')

                    # Prepare Cond
                    cond_ind = random.randint(0, goal_timestep)
                    cond = Image.fromarray(traj_images[cond_ind], mode='RGB')

                    # Prepare Augmentation Params
                    c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
                    j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
                    do_c = np.random.uniform() < crop_prob

                    # Save Augmented Images
                    traj[t]["image_observation"] = aug(img, j, c, do_c)
                    traj[t]["image_context"] = aug(cond, j, c, do_c)
                
                # Save all new observation pairs
                obs.append(np.stack([traj[i]['image_observation'] for i in range(num_images)]))
                envs.append(np.stack([traj[i]['image_context'] for i in range(num_images)]))

                if is_demo:
                    # Save all new affordance pairs
                    goal_obs.append(np.stack([traj[i]['image_observation'] for i in range(goal_timestep, num_images)]))
                    goal_envs.append(np.stack([traj[i]['image_context'] for i in range(goal_timestep, num_images)]))

                # save_image(ptu.from_numpy(goal_obs[-1] / 255.).data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/aug_traj.png")
                # save_image(ptu.from_numpy(goal_envs[-1] / 255.).data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/aug_cond.png")
                # import pdb; pdb.set_trace()

                total_size += num_images

            print("Trajectories:", len(obs))
            print("Datapoints:", total_size)


        # SAVE IMAGES FOR REPRESENTATION TRAINING #
        folder = save_folder # '/media/ashvin/data1/s3doodad/demos/icra2021/outputs_dataset_v4/xx'

        dataset['observations'] = np.expand_dims(np.concatenate(obs, axis=0), 1)
        dataset['env'] = np.concatenate(envs, axis=0)
        train_i, test_i = split_data(dataset['observations'].shape[0])

        train = {'observations': dataset['observations'][train_i], 'env': dataset['env'][train_i]}
        test = {'observations': dataset['observations'][test_i], 'env': dataset['env'][test_i]}

        np.save(folder + '{0}_icra2021_train.npy'.format(save_keyword), train)
        np.save(folder + '{0}_icra2021_test.npy'.format(save_keyword), test)

        # SAVE IMAGES FOR AFFORDANCE TRAINING #
        goal_dataset['observations'] = np.expand_dims(np.concatenate(goal_obs, axis=0), 1)
        goal_dataset['env'] = np.concatenate(goal_envs, axis=0)
        train_i, test_i = split_data(goal_dataset['observations'].shape[0])

        goal_train = {'observations': goal_dataset['observations'][train_i], 'env': goal_dataset['env'][train_i]}
        goal_test = {'observations': goal_dataset['observations'][test_i], 'env': goal_dataset['env'][test_i]}

        np.save(folder + '{0}_icra2021_pixelcnn_train.npy'.format(save_keyword), goal_train)
        np.save(folder + '{0}_icra2021_pixelcnn_test.npy'.format(save_keyword), goal_test)
