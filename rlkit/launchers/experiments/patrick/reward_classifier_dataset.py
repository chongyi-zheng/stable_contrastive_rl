from torch.utils.data import Dataset
import numpy as np

class RewardClassifierConditionalLatentBlockDataset(Dataset):
    """
    Loads latent block dataset
    """

    def __init__(self, data, train=True, cond_on_k_after_done=10, positive_k_before_done=0, flip_network_inputs_randomly=False):
        print('Loading latent block data')
        if train:
            self.data = data['train']
            self.done_indices = data['train_done_indices']
        else:
            self.data = data['test']
            self.done_indices = data['test_done_indices']

        self.size = self.data.shape[0]
        self.traj_length = self.data.shape[1]
        print('Done loading latent block data')

        self.cond_on_k_after_done = cond_on_k_after_done
        self.positive_k_before_done = positive_k_before_done
        self.flip_network_inputs_randomly = flip_network_inputs_randomly

    def __getitem__(self, idx):
        traj_i = idx // self.traj_length
        trans_i = idx % self.traj_length
        done_i = self.done_indices[traj_i].item()
        k_after_done_i = min(self.cond_on_k_after_done + done_i, self.traj_length - 1)

        if trans_i > k_after_done_i:
            trans_i = np.random.choice(k_after_done_i)

        obs = self.data[traj_i, trans_i, :]
        goal = self.data[traj_i, k_after_done_i, :]
        is_close = 1.0 if trans_i + self.positive_k_before_done >= done_i else 0.0

        if self.flip_network_inputs_randomly and np.random.rand() > .5:
            obs, goal = goal, obs
        return np.concatenate((obs, goal, np.float32([is_close])), axis=0)

    def __len__(self):
        return self.size * self.traj_length