import abc
import logging
import numpy as np
from PIL import Image


VALID_IMG_FORMATS = {'CHW', 'CWH', 'HCW', 'HWC', 'WCH', 'WHC'}


class Renderer(metaclass=abc.ABCMeta):
    # TODO: switch to env.render interface
    def __init__(
            self,
            width=48,
            height=48,
            num_channels=3,
            normalize_image=False,
            flatten_image=False,
            create_image_format=None,
            output_image_format='CHW',
    ):
        """Render an image."""
        if output_image_format not in VALID_IMG_FORMATS:
            raise ValueError(
                "Invalid output image format: {}. Valid formats: {}".format(
                    output_image_format, VALID_IMG_FORMATS
                )
            )
        if create_image_format is None:
            create_image_format = output_image_format
        if create_image_format not in VALID_IMG_FORMATS:
            raise ValueError(
                "Invalid input image format: {}. Valid formats: {}".format(
                    create_image_format, VALID_IMG_FORMATS
                )
            )
        if output_image_format != 'CHW':
            logging.warning("An output image format of CHW is recommended, as "
                            "this is the default PyTorch format.")
        self.width = width
        self.height = height
        self.num_channels = num_channels
        self.output_image_format = output_image_format

        self._grayscale = num_channels == 1
        self._normalize_imgs = normalize_image
        self._flatten = flatten_image
        self._create_image_format = create_image_format
        self._letter_to_size = {
            'H': self.height,
            'W': self.width,
            'C': self.num_channels,
        }

    def __call__(self, *args, **kwargs):
        image = self._create_image(*args, **kwargs)
        if self._grayscale:
            image = Image.fromarray(image).convert('L')
            image = np.array(image)
        if self._normalize_imgs:
            image = image / 255.0
        transpose_index = [self._create_image_format.index(c) for c in
                           self.output_image_format]

        image = image.transpose(transpose_index)
        if image.shape != self.image_shape:
            raise RuntimeError("Image shape mismatch: {}, {}".format(
                image.shape,
                self.image_shape,
            ))
        assert image.shape == self.image_shape
        if self._flatten:
            return image.flatten()
        else:
            return image

    @abc.abstractmethod
    def _create_image(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def image_is_normalized(self):
        return self._normalize_imgs

    @property
    def image_shape(self):
        return tuple(
            self._letter_to_size[letter] for letter in self.output_image_format
        )

    @property
    def _create_image_shape(self):
        return tuple(
            self._letter_to_size[letter] for letter in self._create_image_format
        )

    @property
    def image_chw(self):
        return tuple(
            self._letter_to_size[letter] for letter in 'CHW'
        )
