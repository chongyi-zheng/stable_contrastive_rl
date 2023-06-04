import numpy as np
import rlkit.torch.pytorch_util as ptu
import torch

class CountExploration:

    def __init__(self, sae, bins):
        self.sae = sae
        self.n_bins = bins
        self.dim = sae.representation_size

        self.counts = np.ones([self.n_bins for _ in range(self.dim)])


    def increment_counts(self, observations):
        bins = self._observations_to_bins(observations)
        for bin in bins:
            self.counts[bin] += 1

    def get_counts(self, observations):
        bins = self._observations_to_bins(observations)
        return np.array([self.counts[bin] for bin in bins])


    def _observations_to_bins(self, observations):
        latents, _ = self.sae.encode(ptu.np_to_var(observations))
        bins = ptu.get_numpy(torch.round(latents * self.n_bins)).astype(np.uint32)
        return [tuple(bins[i]) for i in range(len(bins))]
