import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.util.io import load_local_or_remote_file
from copy import deepcopy
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize, RandomAffine
import torchvision.transforms.functional as F
import rlkit.torch.pytorch_util as ptu
from torchvision.utils import save_image
import random
import math
import sys
# import cv2

import os

import pickle


# NOTE TO SELF, RUN ONCE WITH ALL, THEN SPLIT (CHANGE NAME) AND RUN  MULTIPLE
# THIS WILL MAKE IT EASIER TO ITERATE ON THE POLICY

AUGMENT = 1 # 5
SIZE = 64

crop_prob = 0.95
cond_frac = 0.15 # Use FIRST cond_frac of traj for conditioning
samples_per_traj = 0 # 25 # Number of goals sampled per traj
filter_keyword = ['fixed'] # Only keep these trajectories
repeat_dict = {'pot': 1, 'drawer': 1, 'tray': 1, 'pnp': 1}

jitter = ColorJitter((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
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
    if do_c: x = F.resized_crop(x, c[0], c[1], c[2], c[3], (SIZE, SIZE), Image.ANTIALIAS)
    else: x = F.resize(x, (SIZE, SIZE), Image.ANTIALIAS)
    x = np.array(j(x)) # / 255
    img = x.transpose([2, 1, 0]).flatten().astype(np.uint8)
    return img

def filter_keys(dictionary, keep=['latent', 'state', 'epsilon']):
    all_keys = list(dictionary.keys())
    for key in all_keys:
        delete = not any([word in key for word in keep])
        if delete: del dictionary[key]

# pretrained_vae_path = path_func("best_vqvae.pt")
# pretrained_vae_path = path_func("itr_1500.pt")
# model = load_local_or_remote_file(pretrained_vae_path)
# ptu.set_gpu_mode(True)
# model.to(ptu.device)

# catagorized_data = {'finetune': []}

# all_files = glob.glob("/media/ashvin/data1/s3doodad/experiments/ashvin/valplus/drawersiql/datacollection/iql-finetune2/run*/id0/video_*_vae.p")
# all_files = filter_files(all_files)
# all_files = repeat_files(all_files)
# print('Predicted Size: ', len(all_files) * 10 * 70 * AUGMENT)
# random.shuffle(all_files)

def make_dataset(all_files, save_folder, pretrained_vae_path, output_name, IMAGE_KEY, extra_keys):
    os.makedirs(save_folder, exist_ok=True)

    total_size = 0
    total_traj = 0
    total_samples = 0

    # model = load_local_or_remote_file(pretrained_vae_path)
    # ptu.set_gpu_mode(True)
    # model.to(ptu.device)

    catagorized_data = {'general': []} # {'finetune': []}
    for key in extra_keys:
        catagorized_data[key] = []

    for filename in all_files:
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
            for _ in range(AUGMENT):

                # Prepare augmentation
                D = deepcopy(data[traj_i])

                traj = D["observations"]
                if len(traj) == 0:
                    continue
                try: img = traj[0][IMAGE_KEY]
                except KeyError: continue

                D['actions'] = np.array(D['actions'])
                D['actions'][:, 3] = np.clip(D['actions'][:, 3], -1, 1)
                # if not (np.all(D['actions'] >= -1) and np.all(D['actions'] <= 1)):
                    # import ipdb; ipdb.set_trace()

                img = img[0:270, 90:570, ::-1]
                img = Image.fromarray(img, mode='RGB')
                c = cropper.get_params(img, (0.9, 1.0), (0.9, 1.1))
                j = jitter.get_params((0.75,1.25), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
                do_c = np.random.uniform() < crop_prob

                # Process images 
                for t in range(len(traj)):
                    if not traj[t]:
                        print(traj_i, t)
                        continue
                    img = traj[t][IMAGE_KEY]
                    img = img[0:270, 90:570, ::-1]
                    img = Image.fromarray(img, mode='RGB')
                    y = aug(img, j, c, do_c)
                    traj[t][IMAGE_KEY] = y
                
                # Encode images
                num_images = len(traj)
                images = np.stack([traj[i][IMAGE_KEY] for i in range(num_images)])
                # latents = model.encode_np(images)

                # Calculate Epsilon
                # dist_list = []
                # for i in range(num_images):
                #     low, high = max(i - eps_steps, 0), min(i + eps_steps, num_images - 1)
                #     for j in range(low, i): dist_list.append(np.linalg.norm(latents[i] - latents[j]))
                #     for j in range(i + 1, high + 1): dist_list.append(np.linalg.norm(latents[i] - latents[j]))
                # avg_dist = np.array(dist_list).mean()
                
                # start_ind = int((1 - goal_frac) * num_images)
                # goal_dist = np.array([np.linalg.norm(latents[i] - latents[-1]) 
                #     for i in range(start_ind, num_images)]).mean()

                # Sample goals
                # if samples_per_traj > 0:
                #     cond_timesteps = int(len(traj) * cond_frac)
                #     num_repeat = math.ceil(samples_per_traj / cond_timesteps)
                #     goal_context = np.repeat(latents[:cond_timesteps], num_repeat, axis=0)[:samples_per_traj]
                #     sampled_goals = model.sample_prior(samples_per_traj, cond=goal_context)

                # # Add latent observations
                # for i in range(num_images):
                #     if samples_per_traj > 0:
                #         traj[i]["presampled_latent_goals"] = sampled_goals[i % samples_per_traj]
                    
                #     traj[i]["initial_latent_state"] = latents[0]
                #     traj[i]["latent_observation"] = latents[i]
                #     traj[i]["latent_achieved_goal"] = latents[i]
                #     traj[i]["latent_desired_goal"] = latents[-1]
                #     #traj[i]["avg_epsilon"] = avg_dist
                #     #traj[i]["goal_epsilon"] = goal_dist
                #     filter_keys(traj[i]) # Delete unnecesary keys

                # decoded_samples = model.decode(ptu.from_numpy(sampled_goals))
                # decoded_traj = model.decode(ptu.from_numpy(latents))
                # save_image(decoded_samples.data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/decoded_samples.png")
                # save_image(ptu.from_numpy(images).data.view(-1, 3, 48, 48).transpose(2, 3),"/home/ashvin/data/sample_testing/traj.png")
                # import pdb; pdb.set_trace()

                # Update
                data_list.append(images)
                total_size += num_images
                total_samples += samples_per_traj
                total_traj += 1

            print("Trajectories:", total_traj)
            print("Datapoints:", total_size)
            print("Samples:", total_samples)

    # SAVE TRAJECTORIES FOR REINFORCEMENT LEARNING #
    for key in catagorized_data.keys():
        data_list = catagorized_data[key]
        save_filename = save_folder + output_name + '_' + key + '.npy'
        print("saving", save_filename, "trajectories", len(data_list))
        if len(data_list) > 0: np.save(save_filename, data_list)
        # if len(data_list) > 0: np.save(path_func(key + '_demos.npy'), data_list)

