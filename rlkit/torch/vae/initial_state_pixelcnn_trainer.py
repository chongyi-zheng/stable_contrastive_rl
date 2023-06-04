import torch
import torch.nn as nn
from torchvision import datasets, transforms
from rlkit.torch import pytorch_util as ptu
from rlkit.util.io import load_local_or_remote_file
from os import path as osp
from sklearn import neighbors
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import time
import os 
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
from rlkit.torch.vae.vq_vae import CVQVAE, VQ_VAE

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()

parser.add_argument("--filepath", type=str)
parser.add_argument("--vaepath", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("-save", action="store_true")
parser.add_argument("-gen_samples", action="store_true", default=True)

parser.add_argument("--dataset",  type=str, default='LATENT_BLOCK',
    help='accepts CIFAR10 | MNIST | FashionMNIST | LATENT_BLOCK')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--img_dim", type=int, default=21)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=1024,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vae(vae_file):
    if vae_file[0] == "/":
        local_path = vae_file
    else:
        local_path = sync_down(vae_file)
    vae = pickle.load(open(local_path, "rb"))
    print("loaded", local_path)
    vae.to("cuda")
    return vae

"""
data loaders
"""

# Load VQVAE + Define Args
vqvae = load_vae(args.vaepath)
root_len = vqvae.root_len
num_embeddings = vqvae.num_embeddings
embedding_dim = vqvae.embedding_dim
imsize = vqvae.imsize
input_channels = vqvae.input_channels
# Load VQVAE + Define Args


dataset_path = '/home/ashvin/data/sasha/demos/33_objects.npy'
new_path = "/home/ashvin/tmp/encoded_multiobj_bullet_data.npy"


# data = load_local_or_remote_file(dataset_path)
# data = data.item()
# del data['env']
# data['env'] = data['observations'][:, 0, :]

# vqvae.to('cpu')
# all_data = []
# for i in tqdm(range(data["observations"].shape[0])):
#     obs = ptu.from_numpy(data["observations"][i] / 255.0 )
#     cond = ptu.from_numpy(data["env"][i] / 255.0 )
#     cond = cond.repeat(obs.shape[0], 1)

#     encodings = vqvae.encode(obs, cond, cont=False)
#     all_data.append(encodings)

# encodings = ptu.get_numpy(torch.cat(all_data, dim=0))
# np.save(new_path, encodings)
# vqvae.to('cuda')

all_data = np.load(new_path, allow_pickle=True)



cond_size = vqvae.num_embeddings

if args.dataset == 'LATENT_BLOCK':
    _, _, train_loader, test_loader, _ = rlkit.torch.vae.pixelcnn_utils.load_data_and_data_loaders(new_path, 'LATENT_BLOCK', args.batch_size)
else:
    train_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=True, download=True,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_Size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        eval('datasets.'+args.dataset)(
            '../data/{}/'.format(args.dataset), train=False,
            transform=transforms.ToTensor(),
        ), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

model = GatedPixelCNN(num_embeddings, root_len**2, args.n_layers, n_classes=vqvae.latent_sizes[1]).to(device)
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

"""
train, test, and log
"""

def cond_train():
    train_loss = []
    for batch_idx, x in enumerate(train_loader):
        start_time = time.time()
        
        #x = (x[:, 0]).cuda()
        x = x.cuda()
        ind_size = vqvae.latent_sizes[0] // vqvae.embedding_dim
        cont_x = vqvae.conditioned_discrete_to_cont(x)
        cond = cont_x[:, vqvae.embedding_dim:].reshape(x.shape[0], -1)
        x = x[:, :ind_size].reshape(-1, root_len, root_len)

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


def cond_test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, x in enumerate(test_loader):
        #x = (x[:, 0]).cuda()

            x = x.cuda()
            ind_size = vqvae.latent_sizes[0] // vqvae.embedding_dim
            cont_x = vqvae.conditioned_discrete_to_cont(x)
            cond = cont_x[:, vqvae.embedding_dim:].reshape(x.shape[0], -1)
            x = x[:, :ind_size].reshape(-1, root_len, root_len)

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

def generate_cond_samples(epoch, batch_size=64):
    data_points = ptu.from_numpy(all_data[np.random.choice(all_data.shape[0], size=(8,))]).long().cuda()
    envs = data_points[:, vqvae.discrete_size:]
    samples = []

    for i in range(8):
        env_latent = data_points[i].reshape(1, -1)

        cont_x = vqvae.conditioned_discrete_to_cont(env_latent)
        cont_cond = cont_x[:, vqvae.embedding_dim:].reshape(1, -1)
        cond_latent = env_latent[:, vqvae.discrete_size:]

        env_image = vqvae.decode(env_latent, cont=False)
        samples.append(env_image)

        e_indices = model.generate(shape=(root_len, root_len),
                batch_size=7, cond=cont_cond.repeat(7, 1)).reshape(-1, root_len**2)

        latents = torch.cat([e_indices, cond_latent.repeat(7, 1)], dim=1)
        samples.append(vqvae.decode(latents, cont=False))

    samples = torch.cat(samples, dim=0)

    save_dir = "/home/ashvin/data/sasha/pixelcnn/vqvae_samples/cond_sample{0}.png".format(epoch)

    save_image(
        samples.data.view(batch_size, input_channels, imsize, imsize).transpose(2, 3),
        save_dir
    )

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, args.epochs):
    vqvae.set_pixel_cnn(model)
    print("\nEpoch {}:".format(epoch))


    cond_train()
    cur_loss = cond_test()

    if args.save or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model")
        pickle.dump(model, open('/home/ashvin/data/sasha/pixelcnn/pixelcnn.pkl', "wb"))
        pickle.dump(vqvae, open('/home/ashvin/data/sasha/pixelcnn/vqvae.pkl', "wb"))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    if args.gen_samples:
        generate_cond_samples(epoch)