import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
from sklearn import neighbors
import numpy as np
from torchvision.utils import save_image
import time
import os 
import pickle
import sys
import utils

# Euclidean distance use tree = neighbors.KDTree(all_data, metric="example"); tree.closest(point)

# 1) Encode entire dataset using BI-GAN
# 2) Load it and put it in the KD tree
# 3) Sample images
# 4) For each sampled image, use the KD tree to find the nearest neighbor
# 5) Ouput the stats of the nearest neighbors (aka mean distance, var, min, max, )


def get_closest_stats(latents):
    #latents = ind_to_cont(latents)
    latents = ptu.get_numpy(latents).reshape(-1, 144)
    all_dists = []
    all_index = []
    for i in range(latents.shape[0]):
        smallest_dist = float('inf')
        index = 0
        for j in range(all_data.shape[0]):
            dist = np.count_nonzero(latents[i]!= all_data[j])
            if dist < smallest_dist:
                smallest_dist = dist
                index = j


        #dist, index = tree.query(latents[i].cpu())
        all_dists.append(smallest_dist)
        all_index.append(index)
    all_dists = np.array(all_dists)
    all_index = np.array(all_index)
    print("Mean:", np.mean(all_dists))
    print("Std:", np.std(all_dists))
    print("Min:", np.min(all_dists))
    print("Max:", np.max(all_dists))

    return torch.LongTensor(all_data[all_index].reshape(-1, 12, 12))