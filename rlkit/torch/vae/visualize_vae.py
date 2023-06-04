import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import tkinter as tk
from rlkit.util.io import sync_down
from rlkit.util.io import load_local_or_remote_file
import torch
import pickle
import numpy as np
import io
import skvideo.io
from PIL import Image, ImageTk
from rlkit.torch import pytorch_util as ptu
from rlkit.data_management.dataset  import \
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset
from rlkit.data_management.images import normalize_image, unnormalize_image

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    # vae = pickle.load(open(local_path, "rb"))
    vae = torch.load(local_path, map_location='cpu')
    print("loaded", local_path)
    vae.to("cpu")
    return vae

class VAEVisualizer(object):
    def __init__(self, path, train_dataset, test_dataset):
        self.path = path
        self.vae = load_vae(path)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = 1

        self.master = tk.Tk()

        self.sliders = []
        for i in range(self.vae.representation_size):
            w = tk.Scale(self.master, from_=-3, to=3, orient=tk.HORIZONTAL, resolution=0.01,)
            x, y = (i % 4), 9 + (i // 4)
            w.grid(row=x, column=y)
            self.sliders.append(w)

        self.new_train_image_button = tk.Button(self.master, text="New Training Image", command=self.new_train_image)
        self.new_train_image_button.grid(row=0, column=8)
        self.new_test_image_button = tk.Button(self.master, text="New Testing Image", command=self.new_test_image)
        self.new_test_image_button.grid(row=1, column=8)
        self.reparametrize_button = tk.Button(self.master, text="Reparametrize", command=self.reparametrize)
        self.reparametrize_button.grid(row=2, column=8)
        self.sweep_button = tk.Button(self.master, text="Sweep", command=self.sweep)
        self.sweep_button.grid(row=3, column=8)
        self.sweep_button = tk.Button(self.master, text="Set Home", command=self.set_home)
        self.sweep_button.grid(row=4, column=12)
        self.sweep_button = tk.Button(self.master, text="Sweep Home", command=self.sweep_home)
        self.sweep_button.grid(row=5, column=12)

        self.leftpanel = tk.Canvas(self.master, width=48, height=48)
        self.rightpanel = tk.Canvas(self.master, width=48, height=48)
        self.leftpanel.grid(row=0, column=0, columnspan=4, rowspan=4)
        self.rightpanel.grid(row=0, column=4, columnspan=4, rowspan=4)

        self.last_mean = np.zeros((self.vae.representation_size))
        self.saved_videos = 0

        # import pdb; pdb.set_trace()
        self.new_train_image()
        # self.sweep()
        self.master.after(0, self.update)

    def load_dataset(filename, test_p=0.9):
        dataset = np.load(filename)
        N = len(dataset)
        n = int(N * test_p)
        train_dataset = dataset[:n, :]
        test_dataset = dataset[n:, :]
        return train_dataset, test_dataset

    def update(self):
        for i in range(self.vae.representation_size):
            self.mean[i] = self.sliders[i].get()
        self.check_change()

        self.master.update()
        self.master.after(10, self.update)

    def get_photo(self, flat_img):
        img = flat_img.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        im = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=im)

    def new_test_image(self):
        self.new_image(True)

    def new_train_image(self):
        self.new_image(False)

    def new_image(self, test=True):
        if test:
            self.batch = self.test_dataset.random_batch(1)
        else:
            self.batch = self.train_dataset.random_batch(1)
        self.sample = self.batch["x_t"]
        #self.sample = self.train_dataset[ind, :] / 255
        img = unnormalize_image(ptu.get_numpy(self.sample).reshape((3, 48, 48)).transpose())
        #img = self.sample.reshape((3, 48, 48)).transpose()
        #img = (255 * img).astype(np.uint8)
        # img = img.astype(np.uint8)
        self.im = Image.fromarray(img)
        #self.leftphoto = ImageTk.PhotoImage(image=self.im)
        #self.leftpanel.create_image(0,0,image=self.leftphoto,anchor=tk.NW)

        self.mu, self.logvar = self.vae.encode(self.sample)
        self.z = self.mu
        self.mean = ptu.get_numpy(self.z).flatten()
        self.home_mean = self.mean.copy()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def reparametrize(self):
        self.z = self.vae.reparameterize((self.mu, self.logvar))
        self.mean = ptu.get_numpy(self.z).flatten()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def check_change(self):
        if not np.allclose(self.mean, self.last_mean):
            z = ptu.from_numpy(self.mean[:, None].transpose())
            self.recon_batch = self.vae.decode(z)[0]
            self.update_reconstruction()
            self.last_mean = self.mean.copy()

    def update_sliders(self):
        for i in range(self.vae.representation_size):
            self.sliders[i].set(self.mean[i])

    def update_reconstruction(self):
        recon_numpy = ptu.get_numpy(self.recon_batch)
        img = recon_numpy.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        # img = img.astype(np.uint8)
        self.rightim = Image.fromarray(img)
        self.rightphoto = ImageTk.PhotoImage(image=self.rightim)
        self.rightpanel.create_image(0,0,image=self.rightphoto,anchor=tk.NW)

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        return ptu.from_numpy(samples)


    def sweep_element(self):
        data = [np.copy(self.mean)]
        # self.rightim.save('/home/ashvin/ros_ws/src/rlkit-private/visualizer/vae/img_0.jpg')
        for i in range(40):
            for k in self.sweep_i:
                if np.random.uniform() < 0.5:
                    sign = 1
                else:
                    sign = -1
                if self.mean[k] >= 3:
                    sign = -1
                if self.mean[k] < -3:
                    sign = 1
                self.mean[k] += sign * 0.25
                self.sliders[k].set(self.mean[k])
            self.check_change()
            self.rightim.save('/home/ashvin/ros_ws/src/railrl-private/visualizer/vae/img_{}.jpg'.format(i + 1))
            data.append(np.copy(self.mean))
            self.master.after(100, self.sweep_element)
        # np.save('/home/ashvin/ros_ws/src/rlkit-private/visualizer/vae/latents.npy', np.array(data))
        self.mean = self.original_mean


    def sweep(self):
        self.original_mean = self.mean.copy()
        self.sweep_i = [i for i in range(self.vae.representation_size)] #temp
        self.sweep_k = 0
        self.master.after(100, self.sweep_element)


    def set_home(self):
        self.home_mean = self.mean.copy()

    def sweep_home(self):
        # decode as we interplote from self.home_mean -> self.mean
        frames = []
        for i, t in enumerate(np.linspace(0, 1, 25)):
            z = t * self.home_mean + (1 - t) * self.mean
            print(t, z)
            z = ptu.from_numpy(z[:, None].transpose())
            recon_batch = self.vae.decode(z)[0]
            recon_numpy = ptu.get_numpy(recon_batch)
            img = recon_numpy.reshape((3, 48, 48)).transpose()
            img = (255 * img).astype(np.uint8)
            frames.append(img)
            # img = img.astype(np.uint8)
            # im = Image.fromarray(img)
            # im.save('tmp/ccvae/0/img_%d.jpg' % i)
        frames += frames[::-1]
        skvideo.io.vwrite("tmp/vae/dog/%d.mp4" % self.saved_videos, frames)
        self.saved_videos += 1


    # def sweep_element(self):
    #     for i in self.sweep_i:
    #         if self.sweep_k > 10:
    #             self.mean[i] = self.original_mean[i]
    #             self.sliders[i].set(self.original_mean[i])
    #             self.check_change()
    #             self.sweep_i += 1
    #             self.sweep_k = 0
    #             self.master.after(100, self.sweep_element)
    #         else:
    #             v = -2.5 + 0.5 * self.sweep_k
    #             self.mean[i] = v
    #             self.sliders[i].set(v)
    #             self.check_change()
    #             self.sweep_k += 1
    #             self.master.after(100, self.sweep_element)
    #     else: # done!
    #         self.mean = self.original_mean

    # def sweep(self):
    #     self.original_mean = self.mean.copy()
    #     self.sweep_i = [1,2,3] #Important latents
    #     self.sweep_k = 0
    #     self.master.after(100, self.sweep_element)

class ConditionalVAEVisualizer(object):
    def __init__(self, path, train_dataset, test_dataset):
        self.path = path
        self.vae = load_vae(path)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = 1

        self.master = tk.Tk()

        self.sliders = []
        for i in range(self.vae.representation_size):
            w = tk.Scale(self.master, from_=-3, to=3, orient=tk.HORIZONTAL, resolution=0.01,)
            x, y = (i % 4), 13 + (i // 4)
            w.grid(row=x, column=y)
            self.sliders.append(w)

        self.new_train_image_button = tk.Button(self.master, text="New Training Image", command=self.new_train_image)
        self.new_train_image_button.grid(row=0, column=12)
        self.new_test_image_button = tk.Button(self.master, text="New Testing Image", command=self.new_test_image)
        self.new_test_image_button.grid(row=1, column=12)
        self.reparametrize_button = tk.Button(self.master, text="Reparametrize", command=self.reparametrize)
        self.reparametrize_button.grid(row=2, column=12)
        self.sweep_button = tk.Button(self.master, text="Sweep", command=self.sweep)
        # self.sweep_button = tk.Button(self.master, text="Sample Prior", command=self.sample_prior)
        self.sweep_button.grid(row=3, column=12)
        self.sweep_button = tk.Button(self.master, text="Set Home", command=self.set_home)
        self.sweep_button.grid(row=4, column=12)
        self.sweep_button = tk.Button(self.master, text="Sweep Home", command=self.sweep_home)
        self.sweep_button.grid(row=5, column=12)

        self.leftleftpanel = tk.Canvas(self.master, width=48, height=48)
        self.leftpanel = tk.Canvas(self.master, width=48, height=48)
        self.rightpanel = tk.Canvas(self.master, width=48, height=48)
        self.leftleftpanel.grid(row=0, column=0, columnspan=4, rowspan=4)
        self.leftpanel.grid(row=0, column=4, columnspan=4, rowspan=4)
        self.rightpanel.grid(row=0, column=8, columnspan=4, rowspan=4)

        self.last_mean = np.zeros((self.vae.representation_size))

        self.saved_videos = 0

        # import pdb; pdb.set_trace()
        self.vae.eval()
        #self.new_train_image()
        self.new_train_image()
        self.master.after(100, self.update)
        # self.sweep()

    def load_dataset(filename, test_p=0.9):
        dataset = np.load(filename).item()

        N = len(dataset["observations"])
        n_random_steps = 100
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        train_dataset = InitialObservationDataset({
            'observations': dataset['observations'][:n, :, :],
        })
        test_dataset = InitialObservationDataset({
            'observations': dataset['observations'][n:, :, :],
        })

        return train_dataset, test_dataset

    def update(self):
        for i in range(self.vae.representation_size):
            self.mean[i] = self.sliders[i].get()
        self.check_change()

        self.master.update()
        self.master.after(10, self.update)

    def get_photo(self, flat_img):
        img = flat_img.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        im = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image=im)

    def new_test_image(self):
        self.new_image(True)

    def new_train_image(self):
        self.new_image(False)

    def new_image(self, test=True):
        if test:
            self.batch = self.test_dataset.random_batch(1)
        else:
            self.batch = self.train_dataset.random_batch(1)
        self.sample = (self.batch["x_t"], self.batch["env"])
        full_x_t = unnormalize_image(ptu.get_numpy(self.sample[0]).reshape((3, 48, 48)).transpose())
        full_x_0 = unnormalize_image(ptu.get_numpy(self.sample[1]).reshape((3, 48, 48)).transpose())
        # img = (255 * img).astype(np.uint8)
        self.im = Image.fromarray(full_x_t)
        self.leftphoto = ImageTk.PhotoImage(image=self.im)
        self.leftpanel.create_image(0,0,image=self.leftphoto,anchor=tk.NW)

        self.x0 = Image.fromarray(full_x_0)
        self.leftleftphoto = ImageTk.PhotoImage(image=self.x0)
        self.leftleftpanel.create_image(0,0,image=self.leftleftphoto,anchor=tk.NW)

        batch = self.sample
        self.mu, self.logvar, self.cond = self.vae.encode(batch[0], batch[1])
        self.z = self.vae.encode(batch[0], batch[1], distrib=False)
        self.mean = ptu.get_numpy(self.z).flatten()
        self.home_mean = self.mean.copy()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def reparametrize(self):
        latent_distribution = (self.mu, self.logvar, self.cond)
        self.z = self.vae.reparameterize(latent_distribution)
        self.mean = ptu.get_numpy(self.z).flatten()
        self.recon_batch = self.vae.decode(self.z)[0]
        self.update_sliders()
        self.check_change()

    def check_change(self):
        if not np.allclose(self.mean, self.last_mean):
            z = ptu.from_numpy(self.mean[:, None].transpose())
            self.recon_batch = self.vae.decode(z)[0]
            self.update_reconstruction()
            self.last_mean = self.mean.copy()

    def update_sliders(self):
        for i in range(self.vae.representation_size):
            self.sliders[i].set(self.mean[i])

    def update_reconstruction(self):
        recon_numpy = ptu.get_numpy(self.recon_batch)
        img = recon_numpy.reshape((3, 48, 48)).transpose()
        img = (255 * img).astype(np.uint8)
        self.rightim = Image.fromarray(img)
        self.rightphoto = ImageTk.PhotoImage(image=self.rightim)
        self.rightpanel.create_image(0,0,image=self.rightphoto,anchor=tk.NW)

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        # if self.normalize:
        #     samples = ((samples - self.train_data_mean) + 1) / 2
        return ptu.from_numpy(samples)

    def sweep_element(self):
        data = [np.copy(self.mean)]
        self.rightim.save('tmp/ccvae/img_0.jpg')
        for i in range(40):
            for k in self.sweep_i:
                if np.random.uniform() < 0.5:
                    sign = 1
                else:
                    sign = -1
                if self.mean[k] >= 3:
                    sign = -1
                if self.mean[k] < -3:
                    sign = 1
                self.mean[k] += sign * 0.25
                self.sliders[k].set(self.mean[k])
            self.check_change()
            self.rightim.save('tmp/ccvae/img_{}.jpg'.format(i + 1))
            data.append(np.copy(self.mean))

    def sweep(self):
        self.original_mean = self.mean.copy()
        self.sweep_i = [i for i in range(self.vae.latent_sizes[0])] #temp
        self.sweep_k = 0
        self.master.after(100, self.sweep_element)


    def set_home(self):
        self.home_mean = self.mean.copy()

    def sweep_home(self):
        # decode as we interplote from self.home_mean -> self.mean
        frames = []
        for i, t in enumerate(np.linspace(0, 1, 25)):
            z = t * self.home_mean + (1 - t) * self.mean
            print(t, z)
            z = ptu.from_numpy(z[:, None].transpose())
            recon_batch = self.vae.decode(z)[0]
            recon_numpy = ptu.get_numpy(recon_batch)
            img = recon_numpy.reshape((3, 48, 48)).transpose()
            img = (255 * img).astype(np.uint8)
            frames.append(img)
            # img = img.astype(np.uint8)
            # im = Image.fromarray(img)
            # im.save('tmp/ccvae/0/img_%d.jpg' % i)
        frames += frames[::-1]
        skvideo.io.vwrite("tmp/vae/dog/%d.mp4" % self.saved_videos, frames)
        self.saved_videos += 1

    def sample_prior(self):
        self.z = self.vae.sample_prior(1, self.batch["env"]) #self.batch["context"]?
        self.recon_batch = self.vae.decode(self.z)[0]
        self.mean = ptu.get_numpy(self.z).flatten()
        self.update_sliders()
        self.check_change()

def load_dataset(filename, test_p=0.9):
        dataset = load_local_or_remote_file(filename).item()

        num_trajectories = dataset["observations"].shape[0]
        n_random_steps = dataset["observations"].shape[1]
        #num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)

        try:
            train_dataset = InitialObservationDataset({
                'observations': dataset['observations'][:n, :, :],
                'env': dataset['env'][:n, :],

            })
            test_dataset = InitialObservationDataset({
                'observations': dataset['observations'][n:, :, :],
                'env': dataset['env'][n:, :],
            })
        except:
            train_dataset = InitialObservationDataset({
                'observations': dataset['observations'][:n, :, :],

            })
            test_dataset = InitialObservationDataset({
                'observations': dataset['observations'][n:, :, :],
            })

        return train_dataset, test_dataset
    # dataset = np.load(filename).item()
    # dataset = dataset.get('observations').reshape((10000, 6912))
    # N = dataset.shape[0]
    # n = int(N * test_p)
    # train_dataset = dataset[:n, :]
    # test_dataset = dataset[n:, :]
    # return train_dataset, test_dataset

dataset = "dog"

if __name__ == "__main__":
    # from rlkit.torch.vae.sawyer2d_push_new_easy_data_wider import generate_vae_dataset
    # train_data, test_data, info = generate_vae_dataset(
    #     N=10000
    # )
    data_path = "/home/ashvin/Desktop/sim_puck_data.npy"
    train_data, test_data = load_dataset(data_path)
    #model_path = "/home/ashvin/data/rail-khazatsky/sasha/cond-rig/hyp-tuning/tuning/run550/id1/vae.pkl"
    model_path = "/home/ashvin/data/sasha/cond-rig/hyp-tuning/dropout/run1/id0/itr_100.pkl"
    ConditionalVAEVisualizer(model_path, train_data, test_data)

    tk.mainloop()
