from rlkit.torch.vae.online_vae_algorithm import OnlineVaeAlgorithm, _train_vae, _test_vae
from rlkit.data_management.online_vae_replay_buffer import OnlineVaeRelabelingBuffer
from rlkit.torch.her.online_vae_her_twin_sac import OnlineVaeHerTwinSac
from rlkit.data_management.images import normalize_image, unnormalize_image
import numpy as np

class DiverseGoals(OnlineVaeHerTwinSac):

    def __init__(
        self,
        p_replace,
        p_add_non_diverse,
        goal_buffer_size=1024,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert self.collection_mode != 'online-parallel', "not sure what happens to sample_goals"
        self.p_replace = p_replace
        self.p_add_non_diverse = p_add_non_diverse
        self.goal_buffer = OnlineVaeRelabelingBuffer(
            self.vae,
            max_size=goal_buffer_size,
            env=self.replay_buffer.env,
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            achieved_goal_key='latent_achieved_goal',
        )
        self.env.goal_sampler = self.sample_goals

    def _post_epoch(self, epoch):
        super()._post_epoch(epoch)
        should_train, amount_to_train = self.vae_training_schedule(epoch)
        rl_start_epoch = int(self.min_num_steps_before_training / self.num_env_steps_per_epoch)
        if should_train or epoch <= (rl_start_epoch - 1):
            self.goal_buffer.refresh_latents(epoch)

    def _handle_path(self, path):
        self.handle_goal_buffer(path)
        super()._handle_path(path)

    def _handle_rollout_ending(self):
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.handle_goal_buffer(path)
        super()._handle_rollout_ending()


    def handle_goal_buffer(self, path):
        """
        Note that we only care about next_obs for goal relabeling.
        """
        next_observations = path['next_observations']
        for next_obs in next_observations:
            self.handle_goal_buffer_step(
                obs=None,
                action=None,
                rewards=None,
                terminal=None,
                next_observation=next_obs,
            )

    def set_goal_buffer_goal(self, idx, next_obs):
        """
        We only keep track of the 'image_observation' and 'latent_observation' of
        next_observation as goals are sampled based on next_observation.
        """
        self.goal_buffer._next_obs['image_observation'][idx] = \
                unnormalize_image(next_obs['image_observation'])
        self.goal_buffer._next_obs['latent_observation'][idx] = \
                next_obs['latent_observation']

    def sample_goals(self, batch_size):
        if self.goal_buffer._size == 0:
            return None
        goal_idxs = self.goal_buffer._sample_indices(batch_size)
        goals = {
            'latent_desired_goal': self.goal_buffer._next_obs['latent_observation'][goal_idxs],
            'image_desired_goal': normalize_image(
                self.goal_buffer._next_obs['image_observation'][goal_idxs]
            )
        }
        return goals

    def handle_goal_buffer_step(self, obs, action, rewards, terminal, next_observation):
        if self.goal_buffer._size < self.goal_buffer.max_size:
            self.set_goal_buffer_goal(self.goal_buffer._size, next_observation)
            self.goal_buffer._size += 1
        else:
            """
            Goal buffer is full. With prob self.p_replace, consider as a goal
            candidate.
            """
            if np.random.random() > self.p_replace:
                return
            """
            Sample random goal for goal buffer and replace if sampled goal is a
            closer neighbor of replay buffer
            """
            goal_idx = self.goal_buffer._sample_indices(1)

            candidate_goal = next_observation['latent_observation']
            goal = self.goal_buffer._next_obs['latent_observation'][goal_idx]
            goal_dist = 0.0
            candidate_dist = 0.0
            for i in range(0, self.goal_buffer._size):
                if i == goal_idx:
                    continue
                cur_goal = self.goal_buffer._next_obs['latent_observation'][i]
                candidate_dist += np.linalg.norm(candidate_goal - cur_goal)
                goal_dist += np.linalg.norm(goal_dist - cur_goal)

            """
            Replace the sampled goal with the candidate goal if sampled goal is
            closer or if prob p_add_non_diverse
            """
            if (
                goal_dist < candidate_dist or
                np.random.random() > self.p_add_non_diverse
            ):
                self.set_goal_buffer_goal(goal_idx, next_observation)
