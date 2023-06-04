import numpy as np
import torch

from rlkit.torch import pytorch_util as ptu


class BeamPlanner(object):

    def __init__(
            self,
            model,
            goal_space='h',

            width=4,
            num_generated_samples=256,
            num_prior_samples=0,

            # width=4,
            # num_generated_samples=0,
            # num_prior_samples=512,

            # width=1,
            # num_generated_samples=0,
            # num_prior_samples=128,

            classifier_thresh=0.6,
            progress_thresh=2.0,
    ):
        # TODO(kuanfang): This class is outdated.
        raise NotImplementedError

        self.vqvae = model['vqvae']
        self.affordance = model['affordance']
        self.classifier = model['classifier']

        self.goal_space = goal_space

        self.width = width
        self.num_generated_samples = num_generated_samples
        self.num_prior_samples = num_prior_samples

        self.classifier_thresh = classifier_thresh
        self.progress_thresh = progress_thresh

    def __call__(self, s_0, s_g, max_steps=5):
        s_t = ptu.from_numpy(s_0[np.newaxis, :, :, :])
        h_t = self.vqvae.encode(s_t, flatten=False)

        # Tile the sequence by the width.
        s_t = s_t.repeat((self.width, 1, 1, 1))
        h_t = h_t.repeat((self.width, 1, 1, 1))

        if self.num_prior_samples > 0:
            assert self.num_prior_candidates > 0
            num_prior_samples = min(self.num_prior_samples,
                                    self.num_prior_candidates)
        else:
            num_prior_samples = 0

        s_g = ptu.from_numpy(s_g[np.newaxis, :, :, :])
        h_g = self.vqvae.encode(s_g, flatten=False)

        # Tile the sequence by the width.
        num_samples_per_beam = self.num_generated_samples + num_prior_samples
        num_samples = self.width * num_samples_per_beam
        s_g = s_g.repeat((num_samples, 1, 1, 1))
        h_g = h_g.repeat((num_samples, 1, 1, 1))

        sequence = [s_t]
        score = np.zeros((self.width,), dtype=np.float32)
        done = np.zeros((self.width,), dtype=np.bool)

        score = ptu.from_numpy(score)
        done = ptu.from_numpy(done.astype(np.uint8))

        for t in range(max_steps):
            print('=============================')
            h_tm1 = h_t
            s_tm1 = s_t

            if self.num_generated_samples > 0:
                z_t = ptu.from_numpy(
                    self.affordance.sample_prior(
                        batch_size=self.width * self.num_generated_samples)
                )
                h_t = self.affordance.decode(
                    z_t,
                    cond=h_tm1.repeat((self.num_generated_samples, 1, 1, 1)),
                ).detach()
                s_t = self.vqvae.decode(h_t).detach()

            if num_prior_samples > 0:
                if num_prior_samples <= self.num_prior_candidates:
                    prior_inds = np.arange(self.num_prior_candidates)
                else:
                    prior_inds = np.random.choice(self.num_prior_candidates,
                                                  num_prior_samples,
                                                  replace=False)

                prior_inds = prior_inds.repeat((self.width, ))
                prior_inds = prior_inds.reshape([-1, self.width])
                prior_inds = prior_inds.transpose([1, 0])
                prior_inds = prior_inds.reshape([-1])
                h_t = self.h_prior[prior_inds]
                s_t = self.s_prior[prior_inds]

                h_t = h_t.view(
                    -1,
                    self.vqvae.embedding_dim,
                    self.vqvae.root_len,
                    self.vqvae.root_len)
                h_tm1 = h_tm1.view(
                    -1,
                    self.vqvae.embedding_dim,
                    self.vqvae.root_len,
                    self.vqvae.root_len)

            # Classify
            c_t = self.classifier(
                h_tm1.repeat((num_samples_per_beam, 1, 1, 1)),
                h_t)
            c_t = torch.sigmoid(c_t)
            c_t = torch.squeeze(c_t, -1)

            if t < max_steps - 1:
                width = self.width
            else:
                width = 1

            if h_t.shape[0] > width:
                score = score.repeat((num_samples_per_beam,))
                done = done.repeat((num_samples_per_beam,))
                done = done.to(torch.uint8)

                dist = self.compute_dist(h_t, h_g).detach()
                dist = torch.where(
                    c_t >= self.classifier_thresh,
                    dist,
                    1e9 * torch.ones_like(dist))

                progress = self.compute_dist(
                    h_t,
                    h_tm1.repeat((num_samples_per_beam, 1, 1, 1)),
                )
                dist = torch.where(
                    progress > self.progress_thresh,
                    dist,
                    1e9 * torch.ones_like(dist))

                print('t: ', t)
                # if t == 0:
                #     dist[width:] = 1e9 * torch.ones_like(dist)[width:]

                score = torch.where(
                    done,
                    score,
                    -dist)

                print('dist: ', dist)
                print('c_t: ', c_t)
                print('done: ', done)

                top_values, top_indices = torch.topk(score, width)

                prev_done = torch.clone(done)
                done = np.logical_or(done, dist < 0.1)
                done = done.to(torch.float32)

                h_t = h_t[top_indices]
                s_t = s_t[top_indices]
                c_t = c_t[top_indices]
                score = score[top_indices]
                progress = progress[top_indices]

                print('== Top ==')
                print('c_t: ', c_t)
                print('score: ', score)
                print('progress: ', progress)

                prev_done = prev_done[top_indices]
                prev_done = prev_done.to(torch.float32)
                done = done[top_indices]

                h_t = torch.where(
                    (prev_done.view(-1, 1, 1, 1) * torch.ones_like(h_t)
                     ).byte(),
                    h_tm1,
                    h_t)
                s_t = torch.where(
                    (prev_done.view(-1, 1, 1, 1) * torch.ones_like(s_t)
                     ).byte(),
                    s_tm1,
                    s_t)

                top_indices //= num_samples_per_beam
                for tau in range(t + 1):
                    sequence[tau] = sequence[tau][top_indices]

            sequence.append(s_t)

        return sequence

    def compute_dist(self, h_t, h_g):
        h_t = torch.flatten(h_t, start_dim=1)
        h_g = torch.flatten(h_g, start_dim=1)
        return torch.sum((h_g - h_t) ** 2, dim=-1)

    def set_and_encode_prior_data(self, prior_data):
        assert prior_data.shape[-1] == self.vqvae.imsize
        assert prior_data.shape[-2] == self.vqvae.imsize
        assert prior_data.shape[-3] == self.vqvae.input_channels

        prior_data = ptu.from_numpy(prior_data)

        self.s_prior = prior_data.view(
            -1,
            self.vqvae.input_channels,
            self.vqvae.imsize,
            self.vqvae.imsize).contiguous()

        self.h_prior = self.vqvae.encode(self.s_prior, flatten=True)

        self.num_prior_candidates = self.s_prior.size()[0]
