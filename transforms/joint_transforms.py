"""
# Code borrowded from:
# https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py
#
#
# MIT License
#
# Copyright (c) 2017 ZijunDeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""


"""
Joint Transform
"""

import math
import numbers
from PIL import Image, ImageOps
import numpy as np
import random
import torchvision.transforms.functional as tf

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        for t in self.transforms:
            img, seg_mask, ood_mask = t(img, seg_mask, ood_mask=ood_mask)

        return img, seg_mask, ood_mask


class RandomCrop(object):
    """
    Take a random crop from the image.

    First the image or crop size may need to be adjusted if the incoming image
    is too small...

    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image

    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """
    def __init__(self, size, ignore_index=0, nopad=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, seg_mask, ood_mask=None, centroid=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        w, h = img.size
        # ASSUME H, W
        th, tw = self.size
        if w == tw and h == th:
            return img, seg_mask, ood_mask

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                seg_mask = ImageOps.expand(seg_mask, border=border, fill=self.ignore_index)
                if ood_mask:
                    ood_mask = ImageOps.expand(ood_mask, border=border, fill=self.ignore_index)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)

        img = img.crop((x1, y1, x1 + tw, y1 + th))
        seg_mask = seg_mask.crop((x1, y1, x1 + tw, y1 + th))
        if ood_mask:
            ood_mask = ood_mask.crop((x1, y1, x1 + tw, y1 + th))


        return img, seg_mask, ood_mask


class ResizeHeight(object):
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.target_h = size
        self.interpolation = interpolation

    def __call__(self, img, seg_mask, ood_mask=None):
        w, h = img.size
        target_w = int(w / h * self.target_h)
        img = img.resize((target_w, self.target_h), self.interpolation)
        seg_mask = seg_mask.resize((target_w, self.target_h), Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize((target_w, self.target_h), Image.NEAREST)

        return img, seg_mask, ood_mask


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        img = img.crop((x1, y1, x1 + tw, y1 + th))
        seg_mask = seg_mask.crop((x1, y1, x1 + tw, y1 + th))
        if ood_mask:
            ood_mask = ood_mask.crop((x1, y1, x1 + tw, y1 + th))
        return img, seg_mask, ood_mask


class CenterCropPad(object):
    def __init__(self, size, ignore_index=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        w, h = img.size
        if isinstance(self.size, tuple):
                tw, th = self.size[0], self.size[1]
        else:
                th, tw = self.size, self.size
        if w < tw:
            pad_x = tw - w
        else:
            pad_x = 0
        if h < th:
            pad_y = th - h
        else:
            pad_y = 0

        if pad_x or pad_y:
            # left, top, right, bottom
            img = ImageOps.expand(img, border=(pad_x, pad_y, pad_x, pad_y), fill=0)
            seg_mask = ImageOps.expand(seg_mask, border=(pad_x, pad_y, pad_x, pad_y),
                                   fill=self.ignore_index)
            if ood_mask:
                ood_mask = ImageOps.expand(ood_mask, border=(pad_x, pad_y, pad_x, pad_y),
                                           fill=self.ignore_index)
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        img = img.crop((x1, y1, x1 + tw, y1 + th))
        seg_mask = seg_mask.crop((x1, y1, x1 + tw, y1 + th))

        if ood_mask:
            ood_mask = ood_mask.crop((x1, y1, x1 + tw, y1 + th))

        return img, seg_mask, ood_mask



class PadImage(object):
    def __init__(self, size, ignore_index):
        self.size = size
        self.ignore_index = ignore_index

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        th, tw = self.size, self.size

        w, h = img.size

        if w > tw or h > th :
            wpercent = (tw/float(w))
            target_h = int((float(img.size[1])*float(wpercent)))
            img = img.resize((tw, target_h), Image.BICUBIC)
            seg_mask = seg_mask.resize((tw, target_h), Image.NEAREST)
            if ood_mask:
                ood_mask = ood_mask.resize((tw, target_h), Image.NEAREST)

        w, h = img.size
        ##Pad
        img = ImageOps.expand(img, border=(0,0,tw-w, th-h), fill=0)
        seg_mask = ImageOps.expand(seg_mask, border=(0,0,tw-w, th-h), fill=self.ignore_index)

        if ood_mask:
            ood_mask = ImageOps.expand(ood_mask, border=(0,0,tw-w, th-h), fill=self.ignore_index)

        return img, seg_mask, ood_mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, seg_mask, ood_mask=None):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg_mask = seg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if ood_mask:
                ood_mask = ood_mask.transpose(Image.FLIP_LEFT_RIGHT)
            return img, seg_mask, ood_mask
        return img, seg_mask, ood_mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        img = img.resize(self.size, Image.BICUBIC)
        seg_mask = seg_mask.resize(self.size, Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize(self.size, Image.NEAREST)

        return img, seg_mask, ood_mask


class Scale(object):
    """
    Scale image such that longer side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, seg_mask, seg_mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)

        img = img.resize((ow, oh), Image.BICUBIC)
        seg_mask = seg_mask.resize((ow, oh), Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize((ow, oh), Image.NEAREST)
        return img, seg_mask, ood_mask


class ScaleMin(object):
    """
    Scale image such that shorter side is == size
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size

        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, seg_mask, ood_mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
        else:
            oh = self.size
            ow = int(self.size * w / h)

        img = img.resize((ow, oh), Image.BICUBIC)
        seg_mask = seg_mask.resize((ow, oh), Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize((ow, oh), Image.NEAREST)
        return img, seg_mask, ood_mask

class Resize(object):
    """
    Resize image to exact size of crop
    """

    def __init__(self, size):
        self.size = (size, size)

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        w, h = img.size
        if (w == h and w == self.size):
            return img, seg_mask, ood_mask
        img = img.resize(self.size, Image.BICUBIC)
        seg_mask = seg_mask.resize(self.size, Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize(self.size, Image.NEAREST)
        return img, seg_mask, ood_mask


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                seg_mask = seg_mask.crop((x1, y1, x1 + w, y1 + h))
                if ood_mask:
                    ood_mask = ood_mask.crop((x1, y1, x1 + w, y1 + h))
                    assert (ood_mask.size == (w, h))
                assert (img.size == (w, h))
                assert (seg_mask.size == (w, h))
                img = img.resize((self.size, self.size), Image.BICUBIC)
                seg_mask = seg_mask.resize((self.size, self.size), Image.NEAREST)
                if ood_mask:
                    ood_mask = ood_mask.resize((self.size, self.size), Image.NEAREST)
                return img, seg_mask, ood_mask

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, seg_mask, ood_mask=ood_mask))


class RandomRotate(object):
    def __init__(self, degree, ignore_index):
        self.degree = degree
        self.ignore_index = ignore_index
        self.pad_color = (0, 0, 0)


    def __call__(self, img, seg_mask, ood_mask=None, centroid=None):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = tf.affine(img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BICUBIC,
                fillcolor=self.pad_color,
                shear=0.0)
        seg_mask = tf.affine(seg_mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=self.ignore_index,
                shear=0.0)
        if ood_mask:
            ood_mask = tf.affine(ood_mask,
                    translate=(0, 0),
                    scale=1.0,
                    angle=rotate_degree,
                    resample=Image.NEAREST,
                    fillcolor=self.ignore_index,
                    shear=0.0)
        return img, seg_mask, ood_mask
        # return img.rotate(rotate_degree, Image.BICUBIC), mask.rotate(
        #     rotate_degree, Image.NEAREST)


class RandomSizeAndCrop(object):
    def __init__(self, size, crop_nopad,
                 scale_min=0.5, scale_max=2.0, ignore_index=0, pre_size=None):
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.pre_size = pre_size

    def __call__(self, img, seg_mask, ood_mask=None, centroid=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size

        # first, resize such that shorter edge is pre_size
        if self.pre_size is None:
            scale_amt = 1.
        elif img.size[1] < img.size[0]:
            scale_amt = self.pre_size / img.size[1]
        else:
            scale_amt = self.pre_size / img.size[0]
        scale_amt *= random.uniform(self.scale_min, self.scale_max)
        w, h = [int(i * scale_amt) for i in img.size]

        if centroid is not None:
            centroid = [int(c * scale_amt) for c in centroid]

        img = img.resize((w, h), Image.BICUBIC)
        seg_mask = seg_mask.resize((w, h), Image.NEAREST)
        if ood_mask:
            ood_mask = ood_mask.resize((w, h), Image.NEAREST)

        return self.crop(img, seg_mask, ood_mask=ood_mask, centroid=centroid)


class SlidingCropOld(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, seg_mask, ood_mask=None):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        seg_mask = np.pad(seg_mask, ((0, pad_h), (0, pad_w)), 'constant',
                          constant_values=self.ignore_label)
        if ood_mask:
            ood_mask = np.pad(ood_mask, ((0, pad_h), (0, pad_w)), 'constant',
                              constant_values=self.ignore_label)
        return img, seg_mask, ood_mask

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        seg_mask = np.array(seg_mask)
        if ood_mask:
            ood_mask = np.array(ood_mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_sublist, seg_mask_sublist = [], []
            if ood_mask:
                ood_mask_sublist = []
            else:
                ood_mask_sublist = None
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    seg_mask_sub = seg_mask[sy: ey, sx: ex]
                    if ood_mask:
                        ood_mask_sub = ood_mask[sy: ey, sx: ex]
                    else:
                        ood_mask_sub = None
                    img_sub, seg_mask_sub, ood_mask_sub = self._pad(img_sub, seg_mask_sub, ood_mask=ood_mask_sub)
                    img_sublist.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    seg_mask_sublist.append(
                        Image.fromarray(
                            seg_mask_sub.astype(
                                np.uint8)).convert('P'))
                    if ood_mask:
                        ood_mask_sublist.append(
                            Image.fromarray(
                                seg_mask_sub.astype(
                                    np.uint8)).convert('P'))
            return img_sublist, seg_mask_sublist, ood_mask_sublist
        else:
            img, seg_mask, ood_mask = self._pad(img, seg_mask, ood_mask=ood_mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            seg_mask = Image.fromarray(seg_mask.astype(np.uint8)).convert('P')
            if ood_mask:
                ood_mask = Image.fromarray(ood_mask.astype(np.uint8)).convert('P')
            return img, seg_mask, ood_mask


class SlidingCrop(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, seg_mask, ood_mask=None):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        seg_mask = np.pad(seg_mask, ((0, pad_h), (0, pad_w)), 'constant',
                          constant_values=self.ignore_label)
        if ood_mask:
            ood_mask = np.pad(ood_mask, ((0, pad_h), (0, pad_w)), 'constant',
                          constant_values=self.ignore_label)

        return img, seg_mask, ood_mask, h, w

    def __call__(self, img, seg_mask, ood_mask=None):
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        seg_mask = np.array(seg_mask)
        if ood_mask:
            ood_mask = np.array(ood_mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_slices, seg_mask_slices, slices_info = [], [], []

            if ood_mask:
                ood_mask_slices = []
            else:
                ood_mask_slices = None

            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    seg_mask_sub = seg_mask[sy: ey, sx: ex]
                    if ood_mask:
                        ood_mask_sub = ood_mask[sy: ey, sx: ex]
                    else:
                        ood_mask_sub = None
                    img_sub, seg_mask_sub, ood_mask_sub, sub_h, sub_w =\
                            self._pad(img_sub, seg_mask_sub, ood_mask_sub)
                    img_slices.append(
                        Image.fromarray(
                            img_sub.astype(
                                np.uint8)).convert('RGB'))
                    seg_mask_slices.append(
                            Image.fromarray(
                                seg_mask_sub.astype(
                                    np.uint8)).convert('P'))
                    if ood_mask:
                        ood_mask_slices.append(
                            Image.fromarray(
                                ood_mask_sub.astype(
                                    np.uint8)).convert('P'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            return img_slices, seg_mask_slices, ood_mask_slices, slices_info
        else:
            img, seg_mask, ood_mask, sub_h, sub_w = self._pad(img, seg_mask, ood_mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            seg_mask = Image.fromarray(seg_mask.astype(np.uint8)).convert('P')
            if ood_mask:
                ood_mask = [Image.fromarray(ood_mask.astype(np.uint8)).convert('P')]

            # ood_mask already contains a value inside a list
            return [img], [seg_mask], ood_mask, [[0, sub_h, 0, sub_w, sub_h, sub_w]]


class ClassUniform(object):
    def __init__(self, size, crop_nopad, scale_min=0.5, scale_max=2.0, ignore_index=0,
                 class_list=[16, 15, 14]):
        """
        This is the initialization for class uniform sampling
        :param size: crop size (int)
        :param crop_nopad: Padding or no padding (bool)
        :param scale_min: Minimum Scale (float)
        :param scale_max: Maximum Scale (float)
        :param ignore_index: The index value to ignore in the GT images (unsigned int)
        :param class_list: A list of class to sample around, by default Truck, train, bus
        """
        self.size = size
        self.crop = RandomCrop(self.size, ignore_index=ignore_index, nopad=crop_nopad)

        self.class_list = class_list.replace(" ", "").split(",")

        self.scale_min = scale_min
        self.scale_max = scale_max

    def detect_peaks(self, image):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)

        :param image: An 2d input images
        :return: Binary output images of the same size as input with pixel value equal
        to 1 indicating that there is peak at that point
        """

        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image == 0)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood,
                                           border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background

        return detected_peaks

    def __call__(self, img, seg_mask, ood_mask=None):
        """
        :param img: PIL Input Image
        :param mask: PIL Input Mask
        :return: PIL output PIL (mask, crop) of self.crop_size
        """
        assert img.size == seg_mask.size
        if ood_mask:
            assert img.size == ood_mask.size

        scale_amt = random.uniform(self.scale_min, self.scale_max)
        w = int(scale_amt * img.size[0])
        h = int(scale_amt * img.size[1])

        if scale_amt < 1.0:
            img, seg_mask = img.resize((w, h), Image.BICUBIC), seg_mask.resize((w, h), Image.NEAREST)
            if ood_mask:
                ood_mask = ood_mask.resize((w, h), Image.NEAREST)
            return self.crop(img, seg_mask, ood_mask)
        else:
            # Smart Crop ( Class Uniform's ABN)
            origw, origh = mask.size
            img_new, seg_mask_new = \
                img.resize((w, h), Image.BICUBIC), seg_mask.resize((w, h), Image.NEAREST)
            if ood_mask:
                ood_mask_new = ood_mask.resize((w, h), Image.NEAREST)
            else:
                ood_mask_new = None
            interested_class = self.class_list  # [16, 15, 14]  # Train, Truck, Bus
            data = np.array(seg_mask)
            arr = np.zeros((1024, 2048))
            for class_of_interest in interested_class:
                # hist = np.histogram(data==class_of_interest)
                map = np.where(data == class_of_interest, data, 0)
                map = map.astype('float64') / map.sum() / class_of_interest
                map[np.isnan(map)] = 0
                arr = arr + map

            origarr = arr
            window_size = 250

            # Given a list of classes of interest find the points on the image that are
            # of interest to crop from
            sum_arr = np.zeros((1024, 2048)).astype('float32')
            tmp = np.zeros((1024, 2048)).astype('float32')
            for x in range(0, arr.shape[0] - window_size, window_size):
                for y in range(0, arr.shape[1] - window_size, window_size):
                    sum_arr[int(x + window_size / 2), int(y + window_size / 2)] = origarr[
                        x:x + window_size,
                        y:y + window_size].sum()
                    tmp[x:x + window_size, y:y + window_size] = \
                        origarr[x:x + window_size, y:y + window_size].sum()

            # Scaling Ratios in X and Y for non-uniform images
            ratio = (float(origw) / w, float(origh) / h)
            output = self.detect_peaks(sum_arr)
            coord = (np.column_stack(np.where(output))).tolist()

            # Check if there are any peaks in the images to crop from if not do standard
            # cropping behaviour
            if len(coord) == 0:
                return self.crop(img_new, seg_mask_new, ood_mask_new)
            else:
                # If peaks are detected, random peak selection followed by peak
                # coordinate scaling to new scaled image and then random
                # cropping around the peak point in the scaled image
                randompick = np.random.randint(len(coord))
                y, x = coord[randompick]
                y, x = int(y * ratio[0]), int(x * ratio[1])
                window_size = window_size * ratio[0]
                cropx = random.uniform(
                    max(0, (x - window_size / 2) - (self.size - window_size)),
                    max((x - window_size / 2), (x - window_size / 2) - (
                        (w - window_size) - x + window_size / 2)))

                cropy = random.uniform(
                    max(0, (y - window_size / 2) - (self.size - window_size)),
                    max((y - window_size / 2), (y - window_size / 2) - (
                        (h - window_size) - y + window_size / 2)))

                return_img = img_new.crop(
                    (cropx, cropy, cropx + self.size, cropy + self.size))
                seg_return_mask = seg_mask_new.crop(
                    (cropx, cropy, cropx + self.size, cropy + self.size))
                if ood_mask:
                    ood_return_mask = ood_mask_new.crop(
                        (cropx, cropy, cropx + self.size, cropy + self.size))
                return (return_img, seg_return_mask, ood_return_mask)
