"""Usage: first download the data (about 12.5GB).
pip install gdown
gdown https://drive.google.com/uc?id=16xlZWvR9Ml0TM0VEgrBGnFbcpzA4uLOY
tar -xvf bair_dataset_numpy.tar.gz

In each numpy file there are the following keys:
data = np.load("train00.npz")
images = data["images"] # T x K x H x 64 x 64 x 3, dtype=np.uint8
actions = data["actions"] # T x H x 4
endeff = data["endeff"] # T x H x 3

where:
T = # of trajectories,
K = camera ID (0 is top, 1 is front),
H = horizon = 15
"""
import numpy as np
import itertools
import torch

from rlkit.data_management.images import normalize_image, unnormalize_image
from rlkit.torch.core import np_to_pytorch_batch
import rlkit.torch.pytorch_util as ptu

from torch.utils import data
from torchvision.transforms import ColorJitter, RandomResizedCrop
import torchvision.transforms as transforms
from PIL import Image

import torchvision.transforms.functional as F

from rlkit.torch.data import BatchLoader, InfiniteBatchLoader
from rlkit.data_management.external.bair_dataset.config import BAIR_DATASET_LOCATION

from rlkit.util.io import sync_down_folder

class BAIRDataset(data.Dataset):
    train_data = {}
    test_data = {}

    def __init__(self,
        is_train,
        dataset_location,
        camera=1,
        n_train_files=10,
        info=None,
        image_size = 64,
        flatten = False,
        transpose = None,
        shift = 0,
        x_crop = 8,
        y_crop = 8,
    ):
        self.is_train = is_train
        self.N = 1024 if is_train else 256
        self.traj_length = 15
        self.camera = camera
        self.image_size = image_size
        self.flatten = flatten
        self.transpose = transpose
        self.shift = shift
        self.x_crop = x_crop
        self.y_crop = y_crop

        # dataset_location = BAIR_DATASET_LOCATION # sync_down_folder(BAIR_DATASET_LOCATION)

        if is_train:
            self.n_files = n_train_files
            for i in range(self.n_files):
                suffix = 'train{:0>2d}.npz'.format(i)
                # filename = BAIR_DATASET_LOCATION + "/" + suffix
                #filename = sync_down_folder(BAIR_DATASET_LOCATION) + "/" + suffix
                filename = dataset_location + "/" + suffix
                data = np.load(filename)
                images = data['images']
                self.train_data[i] = images
        else:
            self.n_files = 1
            suffix = "test.npz"
            # filename = BAIR_DATASET_LOCATION + "/" + suffix
            #filename = sync_down_folder(BAIR_DATASET_LOCATION) + "/" + suffix
            filename = dataset_location + "/" + suffix
            data = np.load(filename)
            images = data['images']
            self.test_data[0] = images

        # print("loaded BAIR dataset:", self.images.shape)
        # self.jitter = ColorJitter((0.7,1.3), (0.95,1.05), (0.95,1.05), (-0.05,0.05))
        # self.crop = RandomResizedCrop((48, 48), (0.9, 0.9), (1, 1))
        # RandomResizedCrop((int(sqrt(self.imlength)), int(sqrt(self.imlength))), (0.9, 0.9), (1, 1))

    def __len__(self):
        return self.N * self.traj_length * self.n_files

    def __getitem__(self, idx):
        i_file = idx // (self.N * self.traj_length)
        data = self.train_data if self.is_train else self.test_data
        images = data[i_file]

        i = idx % (self.N * self.traj_length)
        traj_i = i // self.traj_length
        trans_i = i % self.traj_length

        # idx_out = trans_i + traj_i * self.traj_length + i_file * self.N * self.traj_length
        # print(idx, idx_out, i_file, traj_i, trans_i)

        # x = Image.fromarray(self.data['observations'][traj_i, trans_i].reshape(48, 48, 3), mode='RGB')
        # c = Image.fromarray(self.data['env'][traj_i].reshape(48, 48, 3), mode='RGB')

        # # upsampling gives bad images so random resizing params set to 1 for now
        # # crop = self.crop.get_params(c, (0.9, 0.9), (1, 1))
        # crop = self.crop.get_params(c, (1, 1), (1, 1))
        # jitter = self.jitter.get_params((0.7,1.3), (0.95,1.05), (0.95,1.05), (-0.05,0.05))
        # # jitter = self.jitter.get_params((0.5,1.5), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
        # # jitter = self.jitter.get_params(0.5, 0.1, 0.1, 0.1)

        # x = jitter(F.resized_crop(x, crop[0], crop[1], crop[2], crop[3], (48, 48), Image.BICUBIC))
        # c = jitter(F.resized_crop(c, crop[0], crop[1], crop[2], crop[3], (48, 48), Image.BICUBIC))
        # x_t = normalize_image(np.array(x).flatten()).squeeze()
        # env = normalize_image(np.array(c).flatten()).squeeze()
        step = 64 // self.image_size
        # x_timages = images[traj_i, self.camera, trans_i, 0:64:step, 0:64:step, :]
        x_min = self.x_crop
        x_max = self.x_crop + self.image_size
        y_min = self.y_crop
        y_max = self.y_crop + self.image_size

        x_timages = images[traj_i, self.camera, trans_i, y_min:y_max, x_min:x_max, :]
        env_images = images[traj_i, self.camera, 0, y_min:y_max, x_min:x_max, :]
        if self.transpose != None:
            if type(self.transpose) is bool:
                x_timages = x_timages.transpose()
                env_images = env_images.transpose()
            else:
                x_timages = x_timages.transpose(self.transpose)
                env_images = env_images.transpose(self.transpose)
        if self.flatten:
            x_timages = x_timages.flatten()
            env_images = env_images.flatten()

        x_timages = x_timages / 255.0 - self.shift
        env_images = env_images / 255.0 - self.shift

        data_dict = {
            'x_t': x_timages,
            'env': env_images,
        }
        return data_dict


def generate_dataset(variant):
    dataset_kwargs = variant.get("dataset_kwargs", {})
    train_dataset = BAIRDataset(is_train=True, **dataset_kwargs)
    test_dataset = BAIRDataset(is_train=False, **dataset_kwargs)

    train_batch_loader_kwargs = variant.get(
        'train_batch_loader_kwargs',
        dict(batch_size=32, num_workers=0, )
    )
    test_batch_loader_kwargs = variant.get(
        'test_batch_loader_kwargs',
        dict(batch_size=32, num_workers=0, )
    )

    train_data_loader = data.DataLoader(train_dataset,
        shuffle=True, drop_last=True, **train_batch_loader_kwargs)
    test_data_loader = data.DataLoader(test_dataset,
        shuffle=True, drop_last=True, **test_batch_loader_kwargs)

    train_dataset = InfiniteBatchLoader(train_data_loader)
    test_dataset = InfiniteBatchLoader(test_data_loader)
    info = {}

    return train_dataset, test_dataset, info
