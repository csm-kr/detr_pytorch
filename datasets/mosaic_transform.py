import torch
import random
from PIL import Image
from datasets.transforms_ import RandomCrop, Resize, Compose, RandomSizeCrop


def shift_mosaic_boxes(boxes, shift_x, shift_y):
    boxes[:, 0] = boxes[:, 0] + shift_x
    boxes[:, 1] = boxes[:, 1] + shift_y
    boxes[:, 2] = boxes[:, 2] + shift_x
    boxes[:, 3] = boxes[:, 3] + shift_y
    return boxes


def get_concat_h_cut_center(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, (im1.height - im2.height) // 2))
    return dst


def get_concat_v_cut_center(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, ((im1.width - im2.width) // 2, im1.height))
    return dst


def _load_image_and_targets(index, _load_image, _load_anno, _parse):
    image = _load_image(index)
    anno = _load_anno(index)
    boxes, labels = _parse(anno)
    boxes = torch.FloatTensor(boxes)
    labels = torch.LongTensor(labels)  # 0 ~ 19

    return image, boxes, labels


def load_mosaic(image_id, len_of_dataset, _load_image, _load_anno, _parse, image, boxes, labels):

    if image_id is not None:
        # for coco
        # index1 = index
        index2 = random.randint(0, len_of_dataset - 1)
        index3 = random.randint(0, len_of_dataset - 1)
        index4 = random.randint(0, len_of_dataset - 1)

        # image_id1 = image_id[index1]
        image_id2 = image_id[index2]
        image_id3 = image_id[index3]
        image_id4 = image_id[index4]

    else:
        # for voc
        # index1 = index
        index2 = random.randint(0, len_of_dataset - 1)
        index3 = random.randint(0, len_of_dataset - 1)
        index4 = random.randint(0, len_of_dataset - 1)

        # image_id1 = index1
        image_id2 = index2
        image_id3 = index3
        image_id4 = index4

    image1, boxes1, labels1 = image, boxes, labels
    image2, boxes2, labels2 = _load_image_and_targets(image_id2, _load_image, _load_anno, _parse)
    image3, boxes3, labels3 = _load_image_and_targets(image_id3, _load_image, _load_anno, _parse)
    image4, boxes4, labels4 = _load_image_and_targets(image_id4, _load_image, _load_anno, _parse)

    pre_mosaic_transform = Compose([
        Resize(800, max_size=1333),
        RandomCrop(),
        Resize((400, 400)),
    ])
    size = 400

    image1, boxes1, labels1 = pre_mosaic_transform(image1, boxes1, labels1)
    image2, boxes2, labels2 = pre_mosaic_transform(image2, boxes2, labels2)
    image3, boxes3, labels3 = pre_mosaic_transform(image3, boxes3, labels3)
    image4, boxes4, labels4 = pre_mosaic_transform(image4, boxes4, labels4)

    # bbox 바꾸는 부분
    boxes1 = shift_mosaic_boxes(boxes=boxes1, shift_x=0, shift_y=0)
    boxes2 = shift_mosaic_boxes(boxes=boxes2, shift_x=size, shift_y=0)
    boxes3 = shift_mosaic_boxes(boxes=boxes3, shift_x=0, shift_y=size)
    boxes4 = shift_mosaic_boxes(boxes=boxes4, shift_x=size, shift_y=size)

    # 합치는 부분 - imgae - pil level 에서 합침 refer to https://note.nkmk.me/en/python-pillow-concat-images/
    image12 = get_concat_h_cut_center(image1, image2)
    image34 = get_concat_h_cut_center(image3, image4)

    new_image = get_concat_v_cut_center(image12, image34)
    new_boxes = torch.cat([boxes1, boxes2, boxes3, boxes4], dim=0)
    new_labels = torch.cat([labels1, labels2, labels3, labels4], dim=0)

    return new_image, new_boxes, new_labels