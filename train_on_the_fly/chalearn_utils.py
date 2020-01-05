# -*- coding:utf-8 -*-

import os
import pickle
import random
import sys
import time

import cv2
import numpy as np
import scipy.ndimage


class MultiScaleCornerCrop(object):
    """Crop the given PIL.Image to randomly selected size.
    A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center.
    This crop is finally resized to given size.
    Args:
        scales: cropping scales of the original size
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self,
                 scales,
                 shape,
                 interpolation=cv2.INTER_AREA,
                 crop_positions=('c', 'tl', 'tr', 'bl', 'br')):

        self.scales = scales
        self.shape = shape
        self.interpolation = interpolation
        self.scale = None
        self.crop_position = None
        self.crop_positions = crop_positions

    def __call__(self, img):
        min_length = min(img.shape[0], img.shape[1])
        crop_shape = int(min_length * self.scale)

        image_width = img.shape[0]
        image_height = img.shape[1]
        x1, x2, y1, y2 = 0, 0, 0, 0

        if self.crop_position == 'c':
            center_x = image_width // 2
            center_y = image_height // 2
            box_half = crop_shape // 2
            x1 = center_x - box_half
            y1 = center_y - box_half
            x2 = center_x + box_half
            y2 = center_y + box_half

        elif self.crop_position == 'tl':
            x1 = 0
            y1 = 0
            x2 = crop_shape
            y2 = crop_shape

        elif self.crop_position == 'tr':
            x1 = image_width - crop_shape
            y1 = 0
            x2 = image_width
            y2 = crop_shape

        elif self.crop_position == 'bl':
            x1 = 0
            y1 = image_height - crop_shape
            x2 = crop_shape
            y2 = image_height

        elif self.crop_position == 'br':
            x1 = image_width - crop_shape
            y1 = image_height - crop_shape
            x2 = image_width
            y2 = image_height

        img = img[y1: y2, x1: x2, :]

        return cv2.resize(img, dsize=(self.shape[0], self.shape[1]), interpolation=self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        self.crop_position = self.crop_positions[random.randint(0, len(self.scales) - 1)]


class SpatialElasticDisplacement(object):

    def __init__(self, sigma=2.0, alpha=1.0, order=0, cval=0, mode="constant"):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.cval = cval
        self.mode = mode

    def __call__(self, img):
        if self.p < 0.50:
            image = img
            image_first_channel = image[:, :, 0]
            indices_x, indices_y = self._generate_indices(image_first_channel.shape, alpha=self.alpha, sigma=self.sigma)
            ret_image = (self._map_coordinates(
                image,
                indices_x,
                indices_y,
                order=self.order,
                cval=self.cval,
                mode=self.mode))

            return ret_image
        else:
            return img

    def _generate_indices(self, shape, alpha, sigma):
        assert (len(shape) == 2), "shape: Should be of size 2!"
        dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    def _map_coordinates(self, image, indices_x, indices_y, order=1, cval=0, mode="constant"):
        assert (len(image.shape) == 3), "image.shape: Should be of size 3!"
        result = np.copy(image)
        height, width = image.shape[0:2]
        for c in range(image.shape[2]):
            remapped_flat = scipy.ndimage.interpolation.map_coordinates(
                image[..., c],
                (indices_x, indices_y),
                order=order,
                cval=cval,
                mode=mode
            )
            remapped = remapped_flat.reshape((height, width))
            result[..., c] = remapped
        return result

    def randomize_parameters(self):
        self.p = random.random()
        self.alpha = 1 if random.random() > 0.5 else 2


def data_augmentation(batch, shape):
    mean = [119.65972197, 100.70736938, 107.86672802]
    std = [55.12904017, 52.20730189, 52.71543496]

    mscc = MultiScaleCornerCrop(scales=[1, 0.93, 0.87, 0.80], shape=shape)
    sed = SpatialElasticDisplacement()
    ans_batch = np.zeros((batch.shape[0], batch.shape[1], shape[0], shape[1], 3))
    for video_index in range(batch.shape[0]):
        curr_video = batch[video_index]
        mscc.randomize_parameters()
        sed.randomize_parameters()
        for frame_index in range(curr_video.shape[0]):
            curr_image = curr_video[frame_index]
            normalized_image = (curr_image - mean) / std
            normalized_image = normalized_image / 255.0
            cropped_and_scaled = mscc.__call__(normalized_image)
            ans_batch[video_index, frame_index, :, :, :] = sed.__call__(cropped_and_scaled)

    return ans_batch
