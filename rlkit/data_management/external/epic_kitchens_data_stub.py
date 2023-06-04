import numpy as np
import glob
import skvideo.io
from matplotlib.image import imread

import io
import base64
from IPython.display import HTML
import os.path
from os import path
import pickle

import csv

from matplotlib.image import imread
from torch.utils import data

from rlkit.data_management.images import normalize_image, unnormalize_image

from rlkit.torch import pytorch_util as ptu
# import matplotlib.pyplot as plt

import torchvision

import torchvision.transforms.functional as TF
from PIL import Image

import random
import math

# output_dir = "/private/home/anair17/ashvindev/rlkit/notebooks/outputs/"

# f_action_labels = "/datasets01_101/EPIC_KITCHENS_2018/061218/annotations/EPIC_train_action_labels.csv"
# rows = []
# with open(f_action_labels, 'r') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     i = 0
#     for row in spamreader:
#         rows.append(row)
#         i += 1
# action_labels = rows[1:] # get rid of header

# action_labels_dict = {}
# for row in action_labels:
#     uid = int(row[0])
#     action_labels_dict[uid] = row

# # loads 28470 action sequences

# def get_frame_file(participant_id, video_id, frame_id):
#     frame_string = str(frame_id).zfill(10) 
#     return "/datasets01_101/EPIC_KITCHENS_2018/061218/frames_rgb_flow/rgb/train/%s/%s/frame_%s.jpg" % (participant_id, video_id, frame_string)

# def save_clip(uid, use_cache=True):
#     output_file = output_dir + "clip_%d.mp4" % uid
    
#     row = action_labels_dict[uid]
#     assert uid == int(row[0]), "did not match uid %d %d" % (uid, int(row[0]))
    
#     participant_id = row[1]
#     video_id = row[2]
#     action = row[3]
#     start_frame = int(row[6])
#     end_frame = int(row[7])
    
#     if use_cache and path.exists(output_file):
#         return None
#     else:
#         frames = []

#         for frame in range(start_frame, end_frame):
#             frame_file = get_frame_file(participant_id, video_id, frame)
#             frame = imread(frame_file)
#             frames.append(frame)

#         outputdata = np.array(frames)
#         skvideo.io.vwrite(output_file, outputdata)

#         return frames

# def load_clip(uid, max_frames=-1): # timeit: ~450ms
#     row = action_labels_dict[uid]
#     assert uid == int(row[0]), "did not match uid %d %d" % (uid, int(row[0]))
    
#     participant_id = row[1]
#     video_id = row[2]
#     action = row[3]
#     start_frame = int(row[6])
#     end_frame = int(row[7])
    
#     frames = []

#     if max_frames > 0:
#         idxs = np.linspace(start_frame, end_frame, max_frames).astype(int)
#     else:
#         idxs = range(start_frame, end_frame)

#     for frame in idxs:
#         frame_file = get_frame_file(participant_id, video_id, frame)
#         frame = imread(frame_file)
#         frames.append(frame)

#     outputdata = np.array(frames)

#     return outputdata

# def load_video(uid): # timeit: ~300ms
#     output_file = output_dir + "clip_%d.mp4" % uid
#     videodata = skvideo.io.vread(output_file)
    
#     return videodata

# def show_clip(uid):
#     output_file = output_dir + "clip_%d.mp4" % uid

#     video = io.open(output_file, 'r+b').read()
#     encoded = base64.b64encode(video)
#     return HTML(data='''<video alt="test" controls>
#                     <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#                  </video>'''.format(encoded.decode('ascii')))

# def generate_clips(uids):
#     data = ""
#     for uid in uids:
#         save_clip(uid)
        
#         output_file = output_dir + "clip_%d.mp4" % uid

#         video = io.open(output_file, 'r+b').read()
#         encoded = base64.b64encode(video)
#         data += '''<video alt="test" controls>
#                         <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#                      </video>'''.format(encoded.decode('ascii'))
#     return HTML(data=data)

# def str_in_filter_fn(s):
#     return lambda row: s in row[3]

# def find_label(filter_fn):
#     open_actions = []
#     for row in action_labels:
#         if filter_fn(row):
#             open_actions.append(row)
#     return open_actions

# def search_label(filter_fn, num_videos):
#     open_actions = find_label(filter_fn)
    
#     filt = np.random.choice(open_actions, num_videos)
#     print(filt)
#     ids = [int(row[0]) for row in filt]
    
#     return generate_clips(ids)

# def dataset_stats(rows):
#     L = len(rows)
#     H = np.array([int(row[7]) - int(row[6]) for row in rows])
#     max_H = max(H)
#     min_H = min(H)
#     mean_H = np.mean(H)
#     return (L, min_H, max_H, mean_H)

RANDOM_CROP_X = 16
RANDOM_CROP_Y = 16
WIDTH = 456
HEIGHT = 256
CROP_WIDTH = WIDTH - RANDOM_CROP_X
CROP_HEIGHT = HEIGHT - RANDOM_CROP_Y

def transform_image(img):
    # m = img.shape[1] // 2
    # m0, m1 = m - 120, m + 120
    # img = img[:, m0:m1, :]
    img = img / 255.0
    # img = img - np.array([0.485, 0.456, 0.406])
    # img = img / np.array([0.229, 0.224, 0.225])
    img = img.transpose()
    x, y = np.random.randint(RANDOM_CROP_X), np.random.randint(RANDOM_CROP_Y)
    img = img[:, x:x+CROP_WIDTH, y:y+CROP_HEIGHT]
    return img.flatten()

def transform_batch(img):
    # m = img.shape[2] // 2
    # m0, m1 = m - 120, m + 120
    # img = img[:, 8:248, m0:m1, :]
    img = img.transpose([0, 3, 2, 1])
    x, y = np.random.randint(RANDOM_CROP_X), np.random.randint(RANDOM_CROP_Y)
    return img[:, :, x:x+CROP_WIDTH, y:y+CROP_HEIGHT]

# def viz_rewards(id, savefile):
#     clip_data = epic.load_clip(id)
#     t_clip = epic.transform_batch(clip_data)
#     batch = ptu.from_numpy(t_clip / 255.0)
#     batch = batch.to("cuda")
#     zs = model.encoder(batch)
#     z = ptu.get_numpy(zs)
    
#     z_goal = z[-1, :]
#     distances = []
#     for t in range(len(z)):
#         d = np.linalg.norm(z[t, :] - z_goal)
#         distances.append(d)
    
#     plt.plot(distances)
#     plt.show()
    
#     epic.save_clip(id)
#     return epic.show_clip(id)

def get_clip_as_batch(id, max_frames=-1):
    clip_data = load_clip(id, max_frames)
    t_clip = transform_batch(clip_data)
    batch = ptu.from_numpy(normalize_image(t_clip))
    # batch = batch.to("cuda")
    return batch

def viz_rewards(model, id, savefile=None):
    clip_data = load_clip(id)
    t_clip = transform_batch(clip_data)
    batch = ptu.from_numpy(normalize_image(t_clip))
    batch = batch.to("cuda")

    z = ptu.get_numpy(model.encoder(batch).cpu())
    
    z_goal = z[-1, :]
    distances = []
    for t in range(len(z)):
        d = np.linalg.norm(z[t, :] - z_goal)
        distances.append(d)
    
    plt.figure()
    plt.plot(distances)
    if savefile:
        plt.savefig(savefile)

    return np.array(distances)

def normalize(img):
    return img
    # img = normalize_image(img) # rescale to 0-1
    # img = img - np.array([0.485, 0.456, 0.406])
    # img = img / np.array([0.229, 0.224, 0.225])
    # return img

def get_random_crop_params(img, scale_x, scale_y):
    """Get parameters for ``crop`` for a random sized crop.

    Args:
        img (PIL Image): Image to be cropped.
        scale (tuple): range of size of the origin size cropped
        ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
    """
    w = int(random.uniform(*scale_x) * CROP_WIDTH)
    h = int(random.uniform(*scale_y) * CROP_HEIGHT)

    i = random.randint(0, img.size[1] - h)
    j = random.randint(0, img.size[0] - w)
    
    return i, j, h, w

class EpicTimePredictionDataset(data.Dataset):
    def __init__(self, dataset, output_classes=100):
        self.dataset = dataset
        self.num_traj = len(dataset)
        self.output_classes = output_classes

        self.t_to_pil = torchvision.transforms.ToPILImage()
        self.t_random_resize = torchvision.transforms.RandomResizedCrop(
            size=(CROP_WIDTH, CROP_HEIGHT,),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0), # don't change aspect ratio
        )
        self.t_color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.2, # (0.8, 1.2),
            contrast=0.2, # (0.8, 1.2),
            saturation=0.2, # (0.8, 1.2),
            hue=0.1, # (-0.2, 0.2),
        )
        self.t_to_tensor = torchvision.transforms.ToTensor()

    def load_frame(self, participant_id, video_id, frame_id):
        # img = imread(get_frame_file(participant_id, video_id, frame_id))
        # if img.shape != (256, 456, 3): # (256, 342, 3)
        #     print(participant_id, video_id, frame_id, img.shape)
        # return transform_image(img)

        img = Image.open(get_frame_file(participant_id, video_id, frame_id))
        return img

    def __len__(self):
        return self.num_traj

    def __getitem__(self, index):
        row = self.dataset[index]

        participant_id = row[1]
        video_id = row[2]
        action = row[3]
        start_frame = int(row[6])
        end_frame = int(row[7])
        traj_length = end_frame - start_frame

        d0, dt, dT = sorted(np.random.randint(0, traj_length, 3))
        x0 = self.load_frame(participant_id, video_id, start_frame + d0)
        xt = self.load_frame(participant_id, video_id, start_frame + dt)
        xT = self.load_frame(participant_id, video_id, start_frame + dT + 1)
        yt = int((dt - d0) / (dT + 1 - d0) * self.output_classes)

        # x0 = self.t_to_pil(x0)

        i, j, h, w = get_random_crop_params(
            x0, 
            self.t_random_resize.scale, 
            self.t_random_resize.scale,
        )

        t_color_jitter = self.t_color_jitter.get_params(
            self.t_color_jitter.brightness,
            self.t_color_jitter.contrast,
            self.t_color_jitter.saturation,
            self.t_color_jitter.hue,
        )

        x0 = TF.resized_crop(x0, i, j, h, w, (CROP_HEIGHT, CROP_WIDTH,), self.t_random_resize.interpolation)
        x0 = t_color_jitter(x0)
        x0 = self.t_to_tensor(x0)

        # xt = self.t_to_pil(xt)
        xt = TF.resized_crop(xt, i, j, h, w, (CROP_HEIGHT, CROP_WIDTH,), self.t_random_resize.interpolation)
        xt = t_color_jitter(xt)
        xt = self.t_to_tensor(xt)

        # xT = self.t_to_pil(xT)
        xT = TF.resized_crop(xT, i, j, h, w, (CROP_HEIGHT, CROP_WIDTH,), self.t_random_resize.interpolation)
        xT = t_color_jitter(xT)
        xT = self.t_to_tensor(xT)

        batch = dict(
            # x0=normalize(x0),
            # xt=normalize(xt),
            # xT=normalize(xT),
            x0=x0,
            xt=xt,
            xT=xT,
            yt=yt,
        )

        return batch
