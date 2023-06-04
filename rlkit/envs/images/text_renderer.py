from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np


from rlkit.envs.images import Renderer


class TextRenderer(Renderer):
    """I gave up! See plot_renderer.TextRenderer"""

    def __init__(self, text, *args,
                 text_color='white',
                 background_color='black',
                 **kwargs):
        super().__init__(*args,
                         create_image_format='HWC',
                         **kwargs)

        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf', 100)
        _, h, w = self.image_chw
        self._img = Image.new('RGB', (w, h), background_color)
        self._draw_interface = ImageDraw.Draw(self._img)
        self._draw_interface.text((0, 0), text, fill=text_color, font=font)
        self._np_img = np.array(self._img).copy()

    def _create_image(self, *args, **kwargs):
        return self._np_img
