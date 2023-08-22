import os
import torch
import random
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

import os
import sys
sys.path.append(os.path.abspath('/workspace/detr_pytorch'))

# from utils.util import xy_to_cxcy
from util.box_ops import box_xyxy_to_cxcywh
from utils.download_dataset import download_coco
from datasets.mosaic_transform import load_mosaic


# * COCO_Dataset
class COCO_Dataset(Dataset):
    def __init__(self,
                 data_root='D:\Data\coco',
                 split='train',
                 download=True,
                 transform=None,
                 mosaic_transform=False,
                 boxes_coord='x1y1x2y2',
                 visualization=False):
        '''
        "file structure"
        data_root - train2017
                      - 000000000009.jpg, ...
                  - val2017
                      - 000000000139.jpg, ...
                  - annotations
                      - instance_train2017.json, ...

        :param data_root: coco dataset root
        :param split: the split of dataset 'train' vs 'val'
        :param download: boolean for download coco
        :param transform: transform for detection
        :param mosaic_transform: boolean for mosaic transform
        :param boxes_coord: output boxes coord 'cxywh', x1y1x2y2'
        :param visualization: boolean for download coco
        '''

        super().__init__()

        self.data_root = data_root
        self.split = split
        assert split in ['train', 'val', 'test']
        self.set_name = split + '2017'

        if download:
            download_coco(root_dir=data_root)

        self.transform = transform
        self.mosaic_transform = mosaic_transform
        self.boxes_coord = boxes_coord
        self.visualization = visualization

        self.coco = COCO(os.path.join(self.data_root, 'annotations', 'instances_' + self.set_name + '.json'))
        self.ids = list(sorted(self.coco.imgToAnns.keys()))

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.data_root, self.set_name, path)).convert("RGB")

    def _load_anno(self, id):
        anno = self.coco.loadAnns(ids=self.coco.getAnnIds(imgIds=id))        # anno id 에 해당하는 annotation 을 가져온다.
        return anno

    def __getitem__(self, index):

        image_id = self.ids[index]
        image = self._load_image(image_id)
        anno = self._load_anno(image_id)
        boxes, labels = self.parse_coco(anno)

        w, h = image.size
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        info = {}
        info["image_id"] = torch.tensor([image_id])
        info["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self.mosaic_transform:
            if random.random() > 0.5:
                # load mosaic img
                image, boxes, labels = load_mosaic(image_id=self.ids,
                                                   len_of_dataset=self.__len__(),
                                                   _load_image=self._load_image,
                                                   _load_anno=self._load_anno,
                                                   _parse=self.parse_coco,
                                                   image=image,
                                                   boxes=boxes,
                                                   labels=labels)

        # --------------------------- for transform ---------------------------
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.visualization:
            from utils.visualize_boxes import visualize_boxes_labels
            visualize_boxes_labels(image, boxes, labels, data_type='coco', num_labels=91)

        if self.boxes_coord == 'cxywh':
            boxes = box_xyxy_to_cxcywh(boxes)

        if self.split == "val" or self.split == "test":
            return image, boxes, labels, info
        return image, boxes, labels

    def parse_coco(self, anno, type='bbox', ):
        if type == 'segm':
            return -1

        elif type == 'bbox':
            boxes = []
            labels = []

            for idx, anno_dict in enumerate(anno):

                boxes.append(anno_dict['bbox'])
                labels.append(anno_dict['category_id'])

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            # transform from [x, y, w, h] to [x1, y1, x2, y2]
            boxes[:, 2:] += boxes[:, :2]
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]

            return boxes, labels

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, difficulties, img_name and
        additional_info
        """
        images = list()
        boxes = list()
        labels = list()
        if self.split == "val" or self.split == "test":
            info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            if self.split == "val" or self.split == "test":
                info.append(b[3])

        images = torch.stack(images, dim=0)
        if self.split == "val" or self.split == "test":
            return images, boxes, labels, info
        return images, boxes, labels

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':

    import datasets.transforms_ as T
    device = torch.device('cuda:0')

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    transform_train = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ]),
        ),
        T.Resize((1024, 1024)),
        normalize
    ])

    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.Resize((1024, 1024)),
        normalize
    ])

    coco_dataset = COCO_Dataset(data_root="/usr/src/data/coco",
                                split='train',
                                download=True,
                                transform=transform_train,
                                mosaic_transform=True,
                                boxes_coord='cxywh',
                                visualization=True,)

    images, boxes, labels = coco_dataset.__getitem__(0)
    images, boxes, labels = coco_dataset.__getitem__(1)
    images, boxes, labels = coco_dataset.__getitem__(2)
    print(images)
    print(boxes)
    print(labels)


