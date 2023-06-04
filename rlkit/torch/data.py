import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from rlkit.data_management.images import normalize_image

import rlkit.torch.pytorch_util as ptu

class ImageDataset(Dataset):

    def __init__(self, images, should_normalize=True):
        super().__init__()
        self.dataset = images
        self.dataset_len = len(self.dataset)
        assert should_normalize == (images.dtype == np.uint8)
        self.should_normalize = should_normalize

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idxs):
        samples = self.dataset[idxs, :]
        if self.should_normalize:
            samples = normalize_image(samples)
        return np.float32(samples)


class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source
        self.iter = iter(torch.randperm(len(self.data_source)).tolist())

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = iter(torch.randperm(len(self.data_source)).tolist())
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class InfiniteWeightedRandomSampler(Sampler):

    def __init__(self, data_source, weights):
        assert len(data_source) == len(weights)
        assert len(weights.shape) == 1
        self.data_source = data_source
        # Always use CPU
        self._weights = torch.from_numpy(weights)
        self.iter = self._create_iterator()

    def update_weights(self, weights):
        self._weights = weights
        self.iter = self._create_iterator()

    def _create_iterator(self):
        return iter(
            torch.multinomial(
                self._weights, len(self._weights), replacement=True
            ).tolist()
        )

    def __iter__(self):
        return self

    def __next__(self):
        try:
            idx = next(self.iter)
        except StopIteration:
            self.iter = self._create_iterator()
            idx = next(self.iter)
        return idx

    def __len__(self):
        return 2 ** 62


class BatchLoader:
    def random_batch(self, batch_size):
        raise NotImplementedError


class InfiniteBatchLoader(BatchLoader):
    """Wraps a PyTorch DataLoader"""
    def __init__(self, data_loader):
        self.dataset_loader = data_loader
        self.iterator = iter(self.dataset_loader)

    def __len__(self):
        return len(self.dataset_loader)

    def random_batch(self, batch_size):
        assert batch_size == self.dataset_loader.batch_size
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset_loader)
            batch = next(self.iterator)

        if isinstance(batch, torch.Tensor):
            batch = batch.to(ptu.device)
        elif isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].float().to(ptu.device)

        return batch
