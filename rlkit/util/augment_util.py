"""
torchvision transform wrappers that decouple get_param and forward
This will allow you to use the same augmentation on multiple images
"""

from PIL import Image
import torch
import rlkit.torch.transforms as transforms
import rlkit.torch.transforms.functional as F

# torchvision refactors how they get dimensions every release so now i have to make my own
def get_dimensions(img):
    # returns CHW
    if isinstance(img, torch.Tensor):
        channels = 1 if img.ndim == 2 else img.shape[-3]
        height, width = img.shape[-2:]
    else:
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
    return [channels, height, width]


class RandomResizedCrop(transforms.RandomResizedCrop):
    needs_size = True
    def set_params(self, img):
        self.params = self.get_params(img, self.scale, self.ratio)

    def set_default_params(self, img):
        self.params = None

    def forward(self, img):
        if self.params:
            i, j, h, w = self.params
            img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

        return img


class RandomCrop(transforms.RandomCrop):
    needs_size = True
    def set_params(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        self.params = self.get_params(img, self.size)
    
    def set_default_params(self, img):
        self.params = None
    
    def forward(self, img):
        if self.params:
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            _, height, width = F.get_dimensions(img)
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                img = F.pad(img, padding, self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                img = F.pad(img, padding, self.fill, self.padding_mode)

            i, j, h, w = self.params
            img = F.crop(img, i, j, h, w)

        return img



class ColorJitter(transforms.ColorJitter):
    def set_params(self, img=None):
        self.params = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

    def set_default_params(self, img=None):
        self.params = None

    def forward(self, img):
        if self.params:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.params
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
       
        return img

class Compose(transforms.Compose):
    def set_params(self, img=None):
        for t in self.transforms:
            t.set_params(img)

    def set_default_params(self, img=None):
        for t in self.transforms:
            t.set_default_params(img)


CLASSES = {
    'RandomResizedCrop': RandomResizedCrop,
    'ColorJitter': ColorJitter,
    'RandomCrop': RandomCrop,
}


def create_aug_fn(name, size=None, *args, **kwargs):
    assert name in CLASSES, 'augmentation wrapper class not implemented'
    aug_class = CLASSES[name]
    if getattr(aug_class, 'needs_size', False):
        return aug_class(size, *args, **kwargs)
    else:
        return aug_class(*args, **kwargs)


def create_aug_stack(augment_order, augment_params, size):
    aug_fns = []
    for aug_name in augment_order:
        assert aug_name in augment_params, 'parameters not set for {}'.format(aug_name)
        aug_fns.append(
            create_aug_fn(aug_name, size=size, **augment_params[aug_name])
        )
    return Compose(aug_fns)