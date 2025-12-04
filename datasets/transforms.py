# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target
    
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))
    
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)  

class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))

# nframes = 5 
def crop_five(image, img_0, img_1, img_2, img_3, target, region):
    cropped_image = F.crop(image, *region)
    cropped_image_0 = F.crop(img_0, *region)
    cropped_image_1 = F.crop(img_1, *region)
    cropped_image_2 = F.crop(img_2, *region)
    cropped_image_3 = F.crop(img_3, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, cropped_image_0, cropped_image_1, cropped_image_2, cropped_image_3, target

def hflip_five(image, img_0, img_1, img_2, img_3, target):
    flipped_image = F.hflip(image)
    flipped_image_0 = F.hflip(img_0)
    flipped_image_1 = F.hflip(img_1)
    flipped_image_2 = F.hflip(img_2)
    flipped_image_3 = F.hflip(img_3)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, flipped_image_0, flipped_image_1, flipped_image_2, flipped_image_3, target

class ColorJitter_five(object):
    def __init__(self, brightness=0, contrast=0, saturatio=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness, contrast, saturatio, hue)

    def __call__(self, img, image_0, image_1, image_2, image_3, target):
        return self.color_jitter(img), self.color_jitter(image_0), self.color_jitter(image_1), self.color_jitter(image_2), self.color_jitter(image_3), target

class RandomSizeCrop_five(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, img_0: PIL.Image.Image, img_1: PIL.Image.Image, img_2: PIL.Image.Image, img_3: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        img, img_0, img_1, img_2, img_3, target = crop_five(img, img_0, img_1, img_2, img_3, target, region)
        return img, img_0, img_1, img_2, img_3, target
    
class RandomHorizontalFlip_five(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, img_0, img_1, img_2, img_3, target):
        if random.random() < self.p:
            return hflip_five(img, img_0, img_1, img_2, img_3, target)
        return img, img_0, img_1, img_2, img_3, target
     
class RandomResize_five(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, img_0, img_1, img_2, img_3, target=None):
        size = random.choice(self.sizes)
        img_resize = resize(img, target, size, self.max_size)
        img_0_resize = resize(img_0, None, size, self.max_size)[0]
        img_1_resize = resize(img_1, None, size, self.max_size)[0]
        img_2_resize = resize(img_2, None, size, self.max_size)[0]
        img_3_resize = resize(img_3, None, size, self.max_size)[0]
        return img_resize[0], img_0_resize, img_1_resize, img_2_resize, img_3_resize, img_resize[1]
    
class RandomSelect_five(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, image_0, image_1, image_2, image_3, target):
        if random.random() < self.p:
            return self.transforms1(img, image_0, image_1, image_2, image_3, target)
        return self.transforms2(img, image_0, image_1, image_2, image_3, target)

class Normalize_five(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, image_0, image_1, image_2, image_3, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        image_0 = F.normalize(image_0, mean=self.mean, std=self.std)
        image_1 = F.normalize(image_1, mean=self.mean, std=self.std)
        image_2 = F.normalize(image_2, mean=self.mean, std=self.std)
        image_3 = F.normalize(image_3, mean=self.mean, std=self.std)
        if target is None:
            return image, image_0, image_1, image_2, image_3, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, image_0, image_1, image_2, image_3, target
    
class Compose_five(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, image_0, image_1, image_2, image_3, target):
        for t in self.transforms:
            image, image_0, image_1, image_2, image_3, target = t(image, image_0, image_1, image_2, image_3, target)
        return image, image_0, image_1, image_2, image_3, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class ToTensor_five(object):
    def __call__(self, img, image_0, image_1, image_2, image_3, target):
        return F.to_tensor(img), F.to_tensor(image_0), F.to_tensor(image_1), F.to_tensor(image_2), F.to_tensor(image_3), target
