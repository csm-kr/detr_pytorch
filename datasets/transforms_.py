import torch
import random
from PIL import ImageStat, Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# *** function ***
'''
hflip_(image, boxes, labels)
photometric_distort_(image, boxes, labels)
resize_(image, boxes, labels, size, max_size)
zoom_out_(image, boxes, labels, max_scale)
crop_(image, boxes, labels, region)
'''

# *** class ***
'''
Compose()
RandomSelect()
RandomHorizontalFlip()
RandomPhotoDistortion()
Resize()
RandomResize()
RandomZoomOut()
RandomSizeCrop()
ToTensor()
Normalize()
'''


def hflip_(image, boxes, labels):
    flipped_image = F.hflip(image)
    w, h = image.size
    boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
    return flipped_image, boxes, labels


def photometric_distort_(image, boxes, labels):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.
    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image
    distortions = [F.adjust_brightness,
                   F.adjust_contrast,
                   F.adjust_saturation,
                   F.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if d.__name__ == 'adjust_hue':
            adjust_factor = random.uniform(-18 / 255., 18 / 255.)
        else:
            adjust_factor = random.uniform(0.5, 1.5)
        new_image = d(new_image, adjust_factor)
    return new_image, boxes, labels


def resize_(image,
            boxes,
            labels,
            size,
            max_size=None):
    """
    detection 을 위한 resize function
    :param image: PIL image
    :param boxes: target tensor : [N, 4]
    :param size: resized size (2개 -> resize됨, 1개 -> 작은값 기준으로 resize)
    :param max_size: (1개 & max_size 큰 값 기준으로 resize)
    :return: resize image, scaled boxes
    """
    # 1. get original size
    w, h = image.size

    # ----------check get aspect ratio ------------
    # min_size_ = float(min((h, w)))
    # max_size_ = float(max((h, w)))
    # aspect_ratio = max_size_ / min_size_
    # max = 0
    # if aspect_ratio > max:
    #     max = aspect_ratio
    #     print(aspect_ratio)

    # 2. get resize size
    if isinstance(size, (list, tuple)):
        assert len(size) == 2, 'same size must be a list of 2 elements'
        # 같은값
        size = size
    else:
        # min ~ 1333
        if max_size is not None:
            min_original_size = float(min((h, w)))
            max_original_size = float(max((h, w)))

            # e.g) 800 ~ 1333
            # 작은값을 800으로 맞추었을때의 큰값이 1333 을 넘으면,
            if size / min_original_size * max_original_size > max_size:
                # 큰 값을 1333 으로 맞추었을때의 작은값을 size로 정한다. (더 작아짐)
                size = int(round(max_size / max_original_size * min_original_size))

        # 3. get aspect_ratio
        # 같을 때
        if (w <= h and w == size) or (h <= w and h == size):
            size = (h, w)
        else:
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            size = (oh, ow)

    scaled_image = F.resize(image, size)

    new_h, new_w = size
    new_h, new_w = float(new_h), float(new_w)
    old_h, old_w = h, w
    old_h, old_w = float(old_h), float(old_w)
    ratio_height = new_h / old_h
    ratio_width = new_w / old_w

    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height]).unsqueeze(0)

    return scaled_image, scaled_boxes, labels


def zoom_out_(image, boxes, labels, max_scale):

    original_w, original_h = image.size
    max_scale = max_scale
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    mean_color = tuple(ImageStat.Stat(image)._getmedian())
    left = random.randint(0, new_w - original_w)
    top = random.randint(0, new_h - original_h)

    new_image = Image.new(image.mode, (new_w, new_h), mean_color)
    new_image.paste(image, (left, top))

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)

    return new_image, new_boxes, labels


def crop_(image, boxes, labels, region, min_overlap_ratio=0.5):
    cropped_image = F.crop(image, *region)
    i, j, h, w = region

    # cropped_boxes
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)

    # keep only positive sides of boxes
    keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
    cropped_boxes = cropped_boxes.reshape(-1, 4)

    # keep only upper minimum area overlap ratio
    diff_boxes = boxes[:, 2:] - boxes[:, :2]
    diff_cropped_boxes = cropped_boxes[:, 2:] - cropped_boxes[:, :2]
    keep_area = (diff_cropped_boxes[:, 0] * diff_cropped_boxes[:, 1]) / (diff_boxes[:, 0] * diff_boxes[:, 1]) > min_overlap_ratio
    keep = keep * keep_area

    # cropped_labels
    cropped_labels = labels[keep]
    cropped_boxes = cropped_boxes[keep]

    if len(cropped_boxes) == 0:
        # print("no crop")
        return image, boxes, labels

    return cropped_image, cropped_boxes, cropped_labels


# *** class ***
'''
Compose()
RandomSelect()
RandomHorizontalFlip()
RandomPhotoDistortion()
Resize()
RandomResize()
RandomZoomOut()
RandomSizeCrop()
ToTensor()
Normalize()
'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes, labels):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)
        return image, boxes, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return self.transforms1(image, boxes, labels)
        return self.transforms2(image, boxes, labels)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return hflip_(image, boxes, labels)
        return image, boxes, labels


class RandomPhotoDistortion(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return photometric_distort_(image, boxes, labels)
        return image, boxes, labels


class Resize(object):
    def __init__(self, size, max_size=None):
        """
        ** usage **
        Resize(size=(600, 600))          # same size
        Resize(size=600)                 # keep aspect ratio
        Resize(size=600, max_size=1333)  # keep aspect ratio, but max size is 1333
        """
        self.size = size
        self.max_size = max_size

    def __call__(self, image, boxes, labels):
        return resize_(image=image, boxes=boxes, labels=labels,
                       size=self.size,
                       max_size=self.max_size)


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, image, boxes, labels):
        size = random.choice(self.sizes)
        return resize_(image, boxes, labels, size, self.max_size)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, boxes, labels):
        w = random.randint(min(int(image.width * 0.5), self.min_size), min(image.width, self.max_size))
        h = random.randint(min(int(image.height * 0.5), self.min_size), min(image.height, self.max_size))
        region = T.RandomCrop.get_params(image, [h, w])
        cropped_image, cropped_boxes, cropped_labels = crop_(image, boxes, labels, region)
        return cropped_image, cropped_boxes, cropped_labels


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, boxes, labels):
        w = random.randint(int(image.width * 0.5), image.width)
        h = random.randint(int(image.height * 0.5), image.height)
        region = T.RandomCrop.get_params(image, [h, w])
        if random.random() < self.p:
            return crop_(image, boxes, labels, region)
        return image, boxes, labels


class RandomZoomOut(object):
    def __init__(self, p=0.5, max_scale=3):
        self.p = p
        self.max_scale = max_scale

    def __call__(self, image, boxes, labels):
        if random.random() < self.p:
            return zoom_out_(image, boxes, labels, self.max_scale)
        return image, boxes, labels


class ToTensor(object):
    def __call__(self, image, boxes, labels):
        return F.to_tensor(image), boxes, labels


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, boxes, labels):
        image = F.normalize(image, mean=self.mean, std=self.std)
        h, w = image.shape[-2:]
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        return image, boxes, labels


# def detection_resize_only_image(image, size, max_size):
#
#     h = image.size(1)
#     w = image.size(2)
#
#     # 2. get resize size
#     if isinstance(size, (list, tuple)):
#         size = size
#     else:
#         if max_size is not None:
#             min_original_size = float(min((h, w)))
#             max_original_size = float(max((h, w)))
#
#             # e.g) 800 ~ 1333
#             # 작은값을 800으로 맞추었을때의 큰값이 1333 을 넘으면,
#             if size / min_original_size * max_original_size > max_size:
#                 # 큰 값을 1333 으로 맞추었을때의 작은값을 size로 정한다. (더 작아짐)
#                 size = int(round(max_size / max_original_size * min_original_size))
#
#         # 3. get aspect_ratio
#         if (w <= h and w == size) or (h <= w and h == size):
#             size = (h, w)
#         else:
#             if w < h:
#                 ow = size
#                 oh = int(size * h / w)
#             else:
#                 oh = size
#                 ow = int(size * w / h)
#             size = (oh, ow)
#
#     rescaled_image = F.resize(image, size)
#     return rescaled_image
#
#
# class FRCNNResizeOnlyImage(object):
#     def __init__(self, size, max_size=None):
#         self.size = size
#         self.max_size = max_size
#
#     def __call__(self, image):
#         return detection_resize_only_image(image=image,
#                                            size=self.size,
#                                            max_size=self.max_size)