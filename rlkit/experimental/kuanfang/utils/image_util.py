import numpy as np
from copy import deepcopy
import torch
from PIL import Image

from rlkit.util.augment_util import create_aug_stack

class ImageAugment:

    def __init__(self,
                 image_size=48,
                 augment_order = ['RandomResizedCrop', 'ColorJitter'],
                 # augment_order=['RandomCrop'],  # (chongyiz): use only random crop
                 augment_probability=0.95,
                 augment_params={
                    'RandomResizedCrop': dict(
                        scale=(0.9, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    'ColorJitter': dict(
                        brightness=(0.75, 1.25),
                        contrast=(0.9, 1.1),
                        saturation=(0.9, 1.1),
                        hue=(-0.1, 0.1),
                    ),
                    'RandomCrop': dict(
                        padding=4,
                        padding_mode='edge'
                    ),
                 },
                 ):
        self._image_size = image_size

        self.augment_stack = create_aug_stack(
            augment_order, augment_params, size=(self._image_size, self._image_size)
        )
        self.augment_probability = augment_probability

    def set_augment_params(self, img):
        if torch.rand(1) < self.augment_probability:
            self.augment_stack.set_params(img)
        else:
            self.augment_stack.set_default_params(img)

    def augment(self, img):
        img = self.augment_stack(img)
        return img

    def __call__(self, images, already_tranformed=True):
        if len(images.shape) == 4:
            batched = True
        elif len(images.shape) == 3:
            batched = False
        else:
            raise ValueError

        if already_tranformed:
            images += .5
        
        if self.augment_probability > 0:
            self.set_augment_params(images)
            images = self.augment(images)

        if already_tranformed:
            images -= 0.5

        if not batched:
            images = images[0]
            
        return images
