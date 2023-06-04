import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from os import path as osp
from sklearn import neighbors
import numpy as np
from torchvision.utils import save_image
import time
from torchvision.transforms import ColorJitter, RandomResizedCrop, Resize
from PIL import Image
from rlkit.util.io import load_local_or_remote_file
import os
from tqdm import tqdm
import pickle
import sys
"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())
pixelcnn_dir = sys.path.append(os.getcwd()+ '/pixelcnn')

from rlkit.torch.vae.initial_state_pixelcnn import GatedPixelCNN
import rlkit.torch.vae.pixelcnn_utils
from rlkit.torch.vae.vq_vae import VQ_VAE

"""
Hyperparameters
"""
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--filepath", type=str)
parser.add_argument("--vaepath", type=str)
parser.add_argument("--batch_size", type=int, default=32) #32
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true", default=True)

parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_vae(vae_file, ):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    vae.to('cuda')
    vae.eval()
    return vae

"""
data loaders
"""

# Load VQVAE + Define Args
vqvae = load_vae(args.vaepath)
root_len = vqvae.root_len
num_embeddings = vqvae.num_embeddings
embedding_dim = vqvae.embedding_dim
cond_size = vqvae.num_embeddings
imsize = vqvae.imsize
discrete_size = root_len * root_len
representation_size = embedding_dim * discrete_size
input_channels = vqvae.input_channels
imlength = imsize * imsize * input_channels
# Load VQVAE + Define Args


# Define data loading info
train_path = 'sasha/complex_obj/gr_train_complex_obj_images.npy'
test_path = 'sasha/complex_obj/gr_test_complex_obj_images.npy'
new_path = "/home/ashvin/tmp/encoded_multiobj_bullet_data.npy"

# Define data loading info

def prep_sample_data():
    data = np.load(new_path, allow_pickle=True).item()
    train_data = data['train']#.reshape(-1, discrete_size)
    test_data = data['test']#.reshape(-1, discrete_size)
    return train_data, test_data



def encode_dataset(dataset_path):
    data = load_local_or_remote_file(dataset_path)
    data = data.item()

    all_data = []

    vqvae.to('cpu')
    for i in tqdm(range(data["observations"].shape[0])):
        obs = ptu.from_numpy(data["observations"][i] / 255.0 )
        latent = vqvae.encode(obs, cont=False)
        all_data.append(latent)
    vqvae.to('cuda')

    encodings = ptu.get_numpy(torch.cat(all_data, dim=0))
    return encodings

#### Only run to encode new data ####
train_data = encode_dataset(train_path)
test_data = encode_dataset(test_path)
dataset = {'train': train_data, 'test': test_data}
np.save(new_path, dataset)
#### Only run to encode new data ####
train_data, test_data = prep_sample_data()


_, _, train_loader, test_loader, _ = \
    rlkit.torch.vae.pixelcnn_utils.load_data_and_data_loaders(new_path, 'COND_LATENT_BLOCK', args.batch_size)


model = GatedPixelCNN(num_embeddings, root_len**2, args.n_layers, n_classes=representation_size).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

"""
train, test, and log
"""

def train():
    train_loss = []
    for batch_idx, x in enumerate(train_loader):
        start_time = time.time()
        x_comb = x.cuda()

        cond = vqvae.discrete_to_cont(x_comb[:, vqvae.discrete_size:]).reshape(x.shape[0], -1)
        x = x_comb[:, :vqvae.discrete_size].reshape(-1, root_len, root_len)

        # Train PixelCNN with images
        logits = model(x, cond)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, num_embeddings),
            x.contiguous().view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % args.log_interval == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                args.log_interval * batch_idx / len(train_loader),
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
        #x = (x[:, 0]).cuda()

            x = x.cuda()
            cond = vqvae.discrete_to_cont(x[:, vqvae.discrete_size:]).reshape(x.shape[0], -1)
            x = x[:, :vqvae.discrete_size].reshape(-1, root_len, root_len)

            logits = model(x, cond)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, num_embeddings),
                x.contiguous().view(-1)
            )

            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)

def generate_samples(epoch, test=True, batch_size=64):
    if test:
        dataset = test_data
        dtype = 'test'
    else:
        dataset = train_data
        dtype = 'train'

    rand_indices = np.random.choice(dataset.shape[0], size=(8,))
    data_points = ptu.from_numpy(dataset[rand_indices, 0]).long().cuda()

    samples = []

    for i in range(8):
        env_latent = data_points[i].reshape(1, -1)
        cond = vqvae.discrete_to_cont(env_latent).reshape(1, -1)

        samples.append(vqvae.decode(cond))

        e_indices = model.generate(shape=(root_len, root_len),
                batch_size=7, cond=cond.repeat(7, 1)).reshape(-1, root_len**2)
        samples.append(vqvae.decode(e_indices, cont=False))

    samples = torch.cat(samples, dim=0)
    save_image(
        samples.data.view(batch_size, input_channels, imsize, imsize).transpose(2, 3),
        "/home/ashvin/data/sasha/pixelcnn/vqvae_samples/cond_sample_{0}_{1}.png".format(dtype, epoch)
    )


# def generate_samples(epoch, batch_size=64):
#     num_samples = 8
#     data_points = ptu.from_numpy(all_data[np.random.choice(all_data.shape[0], size=(num_samples,))]).long().cuda()

#     envs = data_points[:, vqvae.discrete_size:]
#     samples = []

#     cond = vqvae.discrete_to_cont(data_points[:, vqvae.discrete_size:]).reshape(num_samples, -1)
#     cond.repeat(num_samples - 1, 1)
#     e_indices = model.generate(
#                 shape=(root_len, root_len),
#                 batch_size=(num_samples - 1) * num_samples,
#                 cond=cond.repeat(num_samples - 1, 1)
#                 )
#     cond_images = vqvae.decode(cond)
#     vqvae.decode(e_indices.reshape(-1, root_len**2), cont=False)

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    vqvae.set_pixel_cnn(model)
    print("\nEpoch {}:".format(epoch))

    model.train()
    train()
    cur_loss = test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model")
        pickle.dump(model, open('/home/ashvin/data/sasha/pixelcnn/pixelcnn.pkl', "wb"))
        pickle.dump(vqvae, open('/home/ashvin/data/sasha/pixelcnn/vqvae.pkl', "wb"))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if args.gen_samples:
        model.eval()
        generate_samples(epoch, test=True)
        generate_samples(epoch, test=False)
