"""
Copy from RAD implementation https://github.com/MishaLaskin/rad
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.experimental.chongyiz.networks.transform_layer import ColorJitterLayer


# def random_crop(imgs, padding=4):
#     n, c, h, w = imgs.shape
#     # padded_imgs = nn.ZeroPad2d(padding)(imgs)
#     padded_imgs = F.pad(imgs, pad=(padding, padding, padding, padding), mode='replicate')
#     _, _, h_padded, w_padded = padded_imgs.shape
#
#     h_crop_max = h_padded - h + 1
#     w_crop_max = w_padded - w + 1
#     h1 = np.random.randint(0, h_crop_max, n)
#     w1 = np.random.randint(0, w_crop_max, n)
#     cropped = torch.empty_like(imgs)
#     for i, (padded_img, w11, h11) in enumerate(zip(padded_imgs, w1, h1)):
#         cropped[i] = padded_img[:, h11:h11 + h, w11:w11 + w]
#
#     return cropped

# (chongyiz): implement random cropping using 2d convolution
def random_crop(imgs, padding=4):
    n, c, h, w = imgs.shape
    assert c % 3 == 0
    padded = F.pad(imgs, pad=(padding, padding, padding, padding), mode='replicate')
    _, _, h_padded, w_padded = padded.shape

    h_crop_max = h_padded - h + 1
    w_crop_max = w_padded - w + 1
    # reference - group 2d convolution:
    #   [1] https://discuss.pytorch.org/t/how-to-apply-different-kernels-to-each-example-in-a-batch-when-using-convolution/84848/4
    #   [2] https://discuss.pytorch.org/t/convolving-a-2d-kernel-on-each-channel/87328/8
    indices = torch.randint(low=0, high=h_crop_max * w_crop_max, size=(n,))
    onehot = ptu.zeros(n, h_crop_max * w_crop_max)
    onehot[torch.arange(n), indices] = 1
    kernel = onehot.reshape(n, 1, h_crop_max, w_crop_max).repeat(
        1, c, 1, 1).reshape(n * c, 1, h_crop_max, w_crop_max)
    # move batch dim into channels
    padded = padded.view(1, -1, h_padded, w_padded)
    cropped = F.conv2d(padded, kernel, stride=1, padding=0, groups=n * c)
    cropped = cropped.view(n, c, h, w)

    return cropped


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3

    imgs = imgs.view([b, frames, 3, h, w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114

    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device)  # broadcast tiling
    return imgs


def random_grayscale(images, p=.3):
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out


def random_cutout(imgs, min_cut=10, max_cut=30):
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        # print(img[:, h11:h11 + h11, w11:w11 + w11].shape)
        cutouts[i] = cut_img
    return cutouts


def random_cutout_color(imgs, min_cut=10, max_cut=30):
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)

    cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
    rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cut_img = img.copy()

        # add random box
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
            rand_box[i].reshape(-1, 1, 1),
            (1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])

        cutouts[i] = cut_img
    return cutouts


def random_flip(images, p=.2):
    # images: [B, C, H, W]
    device = images.device
    bs, channels, h, w = images.shape

    images = images.to(device)

    flipped_images = images.flip([3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1]  # // 3
    images = images.view(*flipped_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)

    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None]

    out = mask * flipped_images + (1 - mask) * images

    out = out.view([bs, -1, h, w])
    return out


def random_rotation(images, p=.3):
    device = images.device
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape

    images = images.to(device)

    rot90_images = images.rot90(1, [2, 3])
    rot180_images = images.rot90(2, [2, 3])
    rot270_images = images.rot90(3, [2, 3])

    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    rnd_rot = np.random.randint(1, 4, size=(images.shape[0],))
    mask = rnd <= p
    mask = rnd_rot * mask
    mask = torch.from_numpy(mask).to(device)

    frames = images.shape[1]
    masks = [torch.zeros_like(mask) for _ in range(4)]
    for i, m in enumerate(masks):
        m[torch.where(mask == i)] = 1
        m = m[:, None] * torch.ones([1, frames]).type(mask.dtype).type(images.dtype).to(device)
        m = m[:, :, None, None]
        masks[i] = m

    out = masks[0] * images + masks[1] * rot90_images + masks[2] * rot180_images + masks[3] * rot270_images

    out = out.view([bs, -1, h, w])
    return out


def random_convolution(imgs):
    _device = imgs.device

    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)

    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)

    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index * batch_size:(trans_index + 1) * batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
    return total_out


def random_color_jitter(imgs):
    b, c, h, w = imgs.shape
    imgs = imgs.view(-1, 3, h, w)
    transform_module = nn.Sequential(ColorJitterLayer(brightness=0.4,
                                                      contrast=0.4,
                                                      saturation=0.4,
                                                      hue=0.5,
                                                      p=1.0,
                                                      batch_size=b))

    imgs = transform_module(imgs).view(b, c, h, w)
    return imgs


def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs


AUG_TO_FUNC = {
    'crop': random_crop,
    'grayscale': random_grayscale,
    'cutout': random_cutout,
    'cutout_color': random_cutout_color,
    'flip': random_flip,
    'rotate': random_rotation,
    'rand_conv': random_convolution,
    'color_jitter': random_color_jitter,
    'translate': random_translate,
}
