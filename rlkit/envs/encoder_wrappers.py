import abc
import numpy as np
import rlkit.torch.pytorch_util as ptu
from gym.spaces import Box, Dict
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.envs.wrappers import ProxyEnv


class Encoder(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode_one_np(self, observation):
        pass

    @property
    @abc.abstractmethod
    def representation_size(self) -> int:
        pass


class AutoEncoder(Encoder, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def decode_one_np(self, observation):
        pass


class ConditionalEncoder(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode_one_np(self, observation, cond):
        pass

    @property
    @abc.abstractmethod
    def representation_size(self) -> int:
        pass


class EncoderWrappedEnv(ProxyEnv):
    def __init__(self,
                 wrapped_env,
                 model: Encoder,
                 step_keys_map=None,
                 reset_keys_map=None,
                 ):
        super().__init__(wrapped_env)
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )

        if step_keys_map is None:
            step_keys_map = {}
        if reset_keys_map is None:
            reset_keys_map = {}
        self.step_keys_map = step_keys_map
        self.reset_keys_map = reset_keys_map
        spaces = self.wrapped_env.observation_space.spaces
        for value in self.step_keys_map.values():
            spaces[value] = latent_space
        for value in self.reset_keys_map.values():
            spaces[value] = latent_space

        self.observation_space = Dict(spaces)
        self.reset_obs = {}

    def step(self, action):
        self.model.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.model.eval()
        for key in self.step_keys_map:
            value = self.step_keys_map[key]
            obs[value] = self.model.encode_one_np(obs[key])
        obs = {**obs, **self.reset_obs}
        return obs

    def reset(self):
        self.model.eval()
        obs = self.wrapped_env.reset()
        for key in self.reset_keys_map:
            value = self.reset_keys_map[key]
            self.reset_obs[value] = self.model.encode_one_np(obs[key])
        obs = self._update_obs(obs)
        return obs

    def get_observation(self):
        obs = self.wrapped_env.get_observation()
        self._update_obs(obs)
        return obs


class PresamplingEncoderWrappedEnv(ProxyEnv):
    def __init__(self,
                 wrapped_env,
                 model: Encoder,
                 step_keys_map=None,
                 reset_keys_map=None,
                 num_sample_on_reset=25,
                 samples_per_trans=1,
                 ):
        super().__init__(wrapped_env)
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )

        if step_keys_map is None:
            step_keys_map = {}
        if reset_keys_map is None:
            reset_keys_map = {}
        self.step_keys_map = step_keys_map
        self.reset_keys_map = reset_keys_map
        spaces = self.wrapped_env.observation_space.spaces
        for value in self.step_keys_map.values():
            spaces[value] = latent_space
        for value in self.reset_keys_map.values():
            spaces[value] = latent_space

        ##### PRESAMPLING CODE #####
        self.num_sample_on_reset = num_sample_on_reset
        self.samples_per_trans = samples_per_trans
        assert samples_per_trans == 1  # Currently supporting just one, but easy change
        presampled_latent_space = Box(
            -10 * np.ones(self.samples_per_trans * model.representation_size),
            10 * np.ones(self.samples_per_trans * model.representation_size),
            dtype=np.float32,
        )
        spaces['presampled_latent_goals'] = presampled_latent_space
        ##### PRESAMPLING CODE #####

        self.observation_space = Dict(spaces)
        self.reset_obs = {}
        self.presampled_goals = None
        self.timestep = None

    def step(self, action):
        self.timestep += 1
        self.model.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.model.eval()
        for key in self.step_keys_map:
            value = self.step_keys_map[key]
            obs[value] = self.model.encode_one_np(obs[key])
        obs = {**obs, **self.reset_obs}
        # obs['presampled_latent_goals'] = self.presampled_goals[self.timestep % self.num_sample_on_reset]
        return obs

    def presample_goals(self, obs):
        # import ipdb; ipdb.set_trace()
        self.model.eval()
        latent_goals = self.model.sample_prior(
            self.num_sample_on_reset, cond=obs['image_observation'], image_cond=True)
        self.presampled_goals = latent_goals

    def reset(self):
        self.timestep = 0
        self.model.eval()
        obs = self.wrapped_env.reset()
        # self.presample_goals(obs)

        for key in self.reset_keys_map:
            value = self.reset_keys_map[key]
            self.reset_obs[value] = self.model.encode_one_np(obs[key])
        obs = self._update_obs(obs)

        return obs


class DictEncoder(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def encode_to_dict_np(self, x: np.ndarray) -> dict:
        pass

    @property
    @abc.abstractmethod
    def output_dict_space(self) -> Dict:
        pass

    def eval(self):
        pass

    def train(self):
        pass


class ConditionalEncoderWrappedEnv(ProxyEnv):
    def __init__(self,
                 wrapped_env,
                 model: ConditionalEncoder,
                 step_keys_map=None,
                 reset_keys_map=None,
                 ):
        super().__init__(wrapped_env)
        self.model = model
        self.representation_size = self.model.representation_size
        latent_space = Box(
            -10 * np.ones(self.representation_size),
            10 * np.ones(self.representation_size),
            dtype=np.float32,
        )

        if step_keys_map is None:
            step_keys_map = {}
        if reset_keys_map is None:
            reset_keys_map = {}
        self.step_keys_map = step_keys_map
        self.reset_keys_map = reset_keys_map
        spaces = self.wrapped_env.observation_space.spaces
        for value in self.step_keys_map.values():
            spaces[value] = latent_space
        for value in self.reset_keys_map.values():
            spaces[value] = latent_space
        self.observation_space = Dict(spaces)
        self.reset_obs = {}

    def step(self, action):
        self.model.eval()
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def _update_obs(self, obs):
        self.model.eval()
        for key in self.step_keys_map:
            value = self.step_keys_map[key]
            obs[value] = self.model.encode_one_np(obs[key], self._initial_img)
        obs = {**obs, **self.reset_obs}
        return obs

    def reset(self):
        self.model.eval()
        obs = self.wrapped_env.reset()
        self._initial_img = obs["image_observation"]
        for key in self.reset_keys_map:
            value = self.reset_keys_map[key]
            self.reset_obs[value] = self.model.encode_one_np(
                obs[key], self._initial_img)
        obs = self._update_obs(obs)
        return obs


class VQVAEWrappedEnv(VAEWrappedEnv):
    def __init__(
        self,
        wrapped_env,
        vae,
        vae_input_key_prefix='image',
        sample_from_true_prior=False,
        decode_goals=False,
        decode_goals_on_reset=True,
        render_goals=False,
        render_rollouts=False,
        reward_params=None,
        goal_sampling_mode="vae_prior",
        num_goals_to_presample=0,
        presampled_goals_path=None,
        imsize=48,
        obs_size=None,
        norm_order=2,
        epsilon=20,
        presampled_goals=None,
    ):
        if reward_params is None:
            reward_params = dict()
        super().__init__(
            wrapped_env,
            vae,
            vae_input_key_prefix,
            sample_from_true_prior,
            decode_goals,
            decode_goals_on_reset,
            render_goals,
            render_rollouts,
            reward_params,
            goal_sampling_mode,
            presampled_goals_path,
            num_goals_to_presample,
            imsize,
            obs_size,
            norm_order,
            epsilon,
            presampled_goals,
        )

        self.representation_size = self.vae.representation_size

        latent_space = Box(
            -10 * np.ones(obs_size or self.representation_size),
            10 * np.ones(obs_size or self.representation_size),
            dtype=np.float32,
        )

        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = latent_space
        spaces['desired_goal'] = latent_space
        spaces['achieved_goal'] = latent_space
        spaces['latent_observation'] = latent_space
        spaces['latent_desired_goal'] = latent_space
        spaces['latent_achieved_goal'] = latent_space
        self.observation_space = Dict(spaces)

    def get_latent_distance(self, latent1, latent2):
        latent1 = ptu.from_numpy(latent1 * self.num_keys).long()
        latent2 = ptu.from_numpy(latent2 * self.num_keys).long()
        return self.vae.get_distance(latent1, latent2)

    def _update_info(self, info, obs):
        self.vae.eval()
        latent_obs = self._encode_one(
            obs[self.vae_input_observation_key])[None]
        latent_goal = self.desired_goal['latent_desired_goal'][None]
        dist = latent_obs - latent_goal
        info["vae_success"] = 1 if np.linalg.norm(
            dist, ord=2) < self.epsilon else 0
        info["vae_dist"] = np.linalg.norm(dist, ord=2)
        info["vae_mdist"] = 0
        info["vae_dist_l1"] = np.linalg.norm(dist, ord=1)
        info["vae_dist_l2"] = np.linalg.norm(dist, ord=2)

    def compute_rewards(self, actions, obs):
        self.vae.eval()
        # TODO: implement log_prob/mdist
        if self.reward_type == 'latent_distance':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            return -dist
        elif self.reward_type == 'latent_sparse':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            success = dist < self.epsilon
            reward = success - 1
            return reward

        elif self.reward_type == 'latent_clamp':
            achieved_goals = obs['latent_achieved_goal']
            desired_goals = obs['latent_desired_goal']
            dist = np.linalg.norm(desired_goals - achieved_goals, axis=1)
            reward = - np.minimum(dist, self.epsilon)
            return reward
        #WARNING: BELOW ARE HARD CODED FOR SIM PUSHER ENV (IN DIMENSION SIZES)
        elif self.reward_type == 'state_distance':
            achieved_goals = obs['state_achieved_goal'].reshape(-1, 4)
            desired_goals = obs['state_desired_goal'].reshape(-1, 4)
            return - np.linalg.norm(desired_goals - achieved_goals, axis=1)
        elif self.reward_type == 'state_sparse':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)
            max_dist = np.linalg.norm(distance, axis=1, ord=np.inf)
            success = max_dist < self.epsilon
            reward = success - 1
            return reward
        elif self.reward_type == 'state_hand_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, :1]
            return - distance
        elif self.reward_type == 'state_puck_distance':
            ob_p = obs['state_achieved_goal'].reshape(-1, 2, 2)
            goal = obs['state_desired_goal'].reshape(-1, 2, 2)
            distance = np.linalg.norm(ob_p - goal, axis=2)[:, 1:]
            return - distance
        elif self.reward_type == 'wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError

    # def _decode(self, latents):
    #     #MAKE INTEGER
    #     self.vae.eval()
    #     latents = ptu.from_numpy(latents * self.num_keys).long()
    #     reconstructions = self.vae.decode(latents)
    #     decoded = ptu.get_numpy(reconstructions)
    #     decoded = np.clip(decoded, 0, 1)
    #     return decoded

    def _decode(self, latents):
        #MAKE INTEGER
        self.vae.eval()
        latents = ptu.from_numpy(latents)
        reconstructions = self.vae.decode(latents, cont=True)
        decoded = ptu.get_numpy(reconstructions)
        decoded = np.clip(decoded, 0, 1)
        return decoded

    def _encode(self, imgs):
        #MAKE FLOAT
        self.vae.eval()
        latents = self.vae.encode(ptu.from_numpy(imgs), cont=True)
        latents = np.array(ptu.get_numpy(latents))
        return latents

    def _reconstruct_img(self, flat_img):
        self.vae.eval()
        img = flat_img.reshape(1, self.input_channels,
                               self.imsize, self.imsize)
        latents = self._encode_one(img)[None]
        imgs = self._decode(latents)
        imgs = imgs.reshape(
            1, self.input_channels, self.imsize, self.imsize
        )
        return imgs[0]

    def _sample_vae_prior(self, batch_size, cont=True):
        self.vae.eval()
        samples = self.vae.sample_prior(batch_size, cont)
        return samples
