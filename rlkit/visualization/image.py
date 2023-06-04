import math

import numpy as np


def add_border(img, border_thickness, border_color):
    imheight, imwidth = img.shape[:2]
    framed_img = np.ones(
        (
            imheight + 2 * border_thickness,
            imwidth + 2 * border_thickness,
            img.shape[2]
        ),
        dtype=np.uint8
    ) * border_color
    framed_img[
        border_thickness:-border_thickness,
        border_thickness:-border_thickness,
        :
    ] = img
    return framed_img


def make_image_fit_into_hwc_format(
        img, output_imwidth, output_imheight, input_image_format
):
    if len(img.shape) == 1:
        if input_image_format == 'HWC':
            hwc_img = img.reshape(output_imheight, output_imwidth, -1)
        elif input_image_format == 'CWH':
            cwh_img = img.reshape(-1, output_imwidth, output_imheight)
            hwc_img = cwh_img.transpose()
        else:
            raise ValueError(input_image_format)
    else:
        a, b, c = img.shape
        transpose_index = [input_image_format.index(channel) for channel in 'HWC']
        hwc_img = img.transpose(transpose_index)

    if hwc_img.shape == (output_imheight, output_imwidth, 3):
        image_that_fits = hwc_img
    else:
        try:
            import cv2
            image_that_fits = cv2.resize(
                hwc_img,
                dsize=(output_imwidth, output_imheight),
            )
        except ImportError:
            image_that_fits = np.zeros((output_imheight, output_imwidth, 3))
            h, w = hwc_img.shape[:2]
            image_that_fits[:h, :w, :] = hwc_img
    return image_that_fits


def combine_images_into_grid(
        imgs, imwidth, imheight,
        max_num_cols=5,
        pad_length=1,
        pad_color=0,
        subpad_length=1,
        subpad_color=127,
        unnormalize=False,
        image_format=None,
        image_formats=None,
):
    if image_formats is None and image_format is None:
        raise RuntimeError(
            "either image_format or image_formats must be provided")
    if image_formats is None:
        image_formats = [image_format for _ in imgs]
    num_imgs = len(imgs)
    num_cols = min(max_num_cols, num_imgs)
    num_rows = int(math.ceil(num_imgs / num_cols))

    new_imgs = []
    for img, image_format in zip(imgs, image_formats):
        img = make_image_fit_into_hwc_format(
            img, imwidth, imheight, image_format)
        if unnormalize:
            img = np.uint8(255 * img)
        if subpad_length > 0:
            img = add_border(img, subpad_length, subpad_color)
        new_imgs.append(img)
    empty_img = np.ones_like(new_imgs[0])

    row_imgs = []

    for row in range(num_rows):
        start_i = row * num_cols
        end_i = min(row * num_cols + num_cols, num_imgs)
        imgs_in_this_row = new_imgs[start_i:end_i].copy()
        imgs_in_this_row += [
            empty_img.copy() for _ in range(num_cols - (end_i - start_i))
        ]
        row_imgs.append(
            np.concatenate(imgs_in_this_row.copy(), axis=1)
        )
    final_image = np.concatenate(row_imgs, axis=0)
    if pad_length > 0:
        final_image = add_border(final_image, pad_length, pad_color)
    return final_image
