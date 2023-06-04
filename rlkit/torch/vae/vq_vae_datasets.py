import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pickle
import numpy as np
from torch.utils.data import Dataset


class BlockDataset(Dataset):
    """
    Creates block dataset of 32X32 images with 3 channels
    requires numpy and cv2 to work
    """

    def __init__(self, file_path, train=True, transform=None):
        print('Loading block data')
        data = np.load(file_path, allow_pickle=True)
        print('Done loading block data')
        data = np.array([cv2.resize(x[0][0][:, :, :3], dsize=(
            32, 32), interpolation=cv2.INTER_CUBIC) for x in data])

        n = data.shape[0]
        cutoff = n//10
        self.data = data[:-cutoff] if train else data[-cutoff:]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        label = 0
        return img, label

    def __len__(self):
        return len(self.data)


class LatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_path, train=True, transform=None, test_p=0.9):
        print('Loading latent block data')
        #self.all_data = np.load(file_path, allow_pickle=True)
        data = np.load(file_path, allow_pickle=True)
        #data = data.reshape(-1, latent_len, latent_len)
        print('Done loading latent block data')
        n = int(data.shape[0] * test_p)
        self.data = data[:n] if train else data[n:]
        self.transform = transform
        self.size = self.data.shape[0]
        self.traj_length = self.data.shape[1]



    def __getitem__(self, idx):
        traj_i = idx // self.traj_length
        trans_i = idx % self.traj_length


        img = self.data[traj_i, trans_i, :]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.size * self.traj_length




class ConditionalLatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, file_path, train=True, transform=None, test_p=0.9):
        print('Loading latent block data')
        data = np.load(file_path, allow_pickle=True)
        data = data.item()

        if train:
            self.data = data['train']
        else:
            self.data = data['test']

        self.size = self.data.shape[0]
        self.traj_length = self.data.shape[1]
        print('Done loading latent block data')
        self.transform = transform



    def __getitem__(self, idx):
        traj_i = idx // self.traj_length
        trans_i = idx % self.traj_length #1
        
        obs = self.data[traj_i, trans_i, :]
        cond = self.data[traj_i, 0, :]


        img = np.concatenate([obs, cond], axis=0)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return self.size * self.traj_length
