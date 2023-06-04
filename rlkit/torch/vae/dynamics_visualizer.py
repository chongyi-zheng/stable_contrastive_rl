import numpy as np
from rlkit.data_management.images import normalize_image
from rlkit.torch import pytorch_util as ptu
from multiworld.envs.pygame.point2d import Point2DWallEnv
import pickle
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in
from multiworld.core.image_env import ImageEnv, normalize_image
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from torchvision.utils import save_image
import torch

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    vae.to("cpu")
    return vae

class DynamicsVisualizer:
	def __init__(self, path, episode_len, num_episodes, env=None):
		self.path = path
		self.vae = load_vae(path)
		self.env = env
		self.episode_len = episode_len
		self.num_episodes = num_episodes
		self.data = []
		self.run_experiment()


	def run_experiment(self):
		all_imgs = []
		policy = OUStrategy(env.action_space)
		for i in range(self.num_episodes):
			state = self.env.reset()
			img = ptu.from_numpy(state['image_observation']).view(1, 6912)
			latent_state = self.vae.encode(img)[0]

			true_curr = state['image_observation'] * 255.0
			all_imgs.append(ptu.from_numpy(true_curr).view(3, 48, 48))

			actions = []
			for j in range(self.episode_len):
				u = policy.get_action_from_raw_action(env.action_space.sample())
				actions.append(u)
				state = self.env.step(u)[0]
				true_curr = state['image_observation'] * 255.0
				all_imgs.append(ptu.from_numpy(true_curr).view(3, 48, 48))

			pred_curr = self.vae.decode(latent_state)[0] * 255.0
			all_imgs.append(pred_curr.view(3, 48, 48))

			for j in range(self.episode_len):
				u = ptu.from_numpy(actions[j]).view(1, 2)
				latent_state = self.vae.process_dynamics(latent_state, u)
				pred_curr = self.vae.decode(latent_state)[0] * 255.0
				all_imgs.append(pred_curr.view(3, 48, 48))

		all_imgs = torch.stack(all_imgs)
		save_image(
	        all_imgs.data,
	        "/home/khazatsky/rail/data/rail-khazatsky/sasha/dynamics_visualizer/dynamics.png",
	        nrow=self.episode_len + 1,
	    )

if __name__ == "__main__":
	env = Point2DWallEnv(
			render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,)


	wrapped_env = ImageEnv(
					env,
					48,
					init_camera=sawyer_init_camera_zoomed_in,
					transpose=True,
					normalize=True,
					non_presampled_goal_img_is_garbage=False,
				)
	model_path = "/home/khazatsky/rail/data/rail-khazatsky/sasha/dynamics/local-dynamics-pointmass/run1/id0/itr_500.pkl"
	DynamicsVisualizer(model_path, 10, 10, env=wrapped_env)
