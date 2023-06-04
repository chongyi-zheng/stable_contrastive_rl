"""Numpy-based image augmentation."""
import random

import numpy as np


class BatchPad(object):
    def __init__(self, image_format, h_pad, w_pad, mode='edge'):
        self.image_format = image_format
        self.h_pad = h_pad
        self.w_pad = w_pad
        self.pads = [(0, 0), None, None, None]
        self.pads[1 + image_format.index('H')] = (h_pad, h_pad)
        self.pads[1 + image_format.index('W')] = (w_pad, w_pad)
        self.pads[1 + image_format.index('C')] = (0, 0)
        self.mode = mode

    def __call__(self, img):
        return np.pad(img, self.pads, mode=self.mode)


class JointRandomCrop(object):
    """Based on `torchvision.transformations.RandomCrop`."""

    def __init__(
            self,
            image_format,
            output_size,
    ):
        super().__init__()
        self.image_format = image_format
        self.output_hw = (
            output_size[image_format.index('H')],
            output_size[image_format.index('W')],
        )

    def __call__(self, img1, img2):
        y0, x0, h, w = self._get_params(img1, self.output_hw)

        return (
            self._crop(img1, y0, x0, h, w),
            self._crop(img2, y0, x0, h, w),
        )

    def _crop(self, img: np.ndarray, y0: int, x0: int, h: int, w: int):
        """Crop the given image.

        Args:
            img: Image to be cropped.
            y0: starting y-coordinate
            x0: starting x-coordinate
            h: Height of the cropped image.
            w: Width of the cropped image.

        Returns:
            Cropped image.
        """
        if self.image_format[0] == 'C':
            cropped = img[..., y0:y0 + h, x0:x0 + w]
        elif self.image_format[1] == 'C':
            cropped = img[..., y0:y0 + h, :, x0:x0 + w]
        elif self.image_format[2] == 'C':
            cropped = img[..., y0:y0 + h, x0:x0 + w, :]
        else:
            raise ValueError(self.image_format)
        return cropped

    def _get_params(self, img, output_hw):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_hw (tuple): Expected output size of the crop.

        Returns:
            tuple: params (y0, x0, h, w) to be passed to ``crop`` for random crop.
        """

        h, w = self._get_hw(img)
        th, tw = output_hw
        if w == tw and h == th:
            return 0, 0, h, w

        y0 = random.randint(0, h - th)
        x0 = random.randint(0, w - tw)
        return y0, x0, th, tw

    def _get_hw(self, img):
        shape = img.shape
        h_index = 1 + self.image_format.index('H')
        w_index = 1 + self.image_format.index('W')
        return shape[h_index], shape[w_index]

    def __repr__(self):
        return self.__class__.__name__ + '(output_hw={0})'.format(
            self.output_hw
        )
