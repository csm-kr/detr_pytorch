import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from xml.etree.ElementTree import parse
from datasets.mosaic_transform import load_mosaic
from utils.download_dataset import download_voc
from utils.util import xy_to_cxcy


class VOC_Dataset(Dataset):

    # not background for coco
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

    """
    "file structure"
    data_root - VOCtest_06-Nov-2007
                  - VOCdevkit
                      - VOC2007
                          - JPEGImages
                          - Annotations
                          - ...
              - VOCtrainval_06-Nov-2007
                  - VOCdevkit
                      - VOC2007
                          - JPEGImages
                          - Annotations
                          - ...
              - VOCtrainval_11-May-2012
                  - VOCdevkit
                      - VOC2012
                          - JPEGImages
                          - Annotations
                          - ...
    """
    def __init__(self,
                 data_root='D:\data\\voc',
                 split='train',
                 download=True,
                 transform=None,
                 mosaic_transform=False,
                 boxes_coord='x1y1x2y2',
                 visualization=True):
        super(VOC_Dataset, self).__init__()

        self.data_root = data_root
        assert split in ['train', 'test']
        self.split = split

        self.download = download
        if self.download:
            download_voc(root_dir=data_root)

        self.transform = transform
        self.mosaic_transform = mosaic_transform
        self.boxes_coord = boxes_coord
        self.visualization = visualization

        # -------------------------- data setting --------------------------
        self.data_list = []
        self.img_list = []
        self.anno_list = []

        for i in os.listdir(self.data_root):
            if i.find('.tar') == -1 and i.find(self.split) != -1:  # split 이 포함된 data - except .tar 제외
                self.data_list.append(i)

        for data_ in self.data_list:

            self.img_list.extend(glob.glob(os.path.join(os.path.join(self.data_root, data_), '*/*/JPEGImages/*.jpg')))
            self.anno_list.extend(glob.glob(os.path.join(os.path.join(self.data_root, data_), '*/*/Annotations/*.xml')))
            # only voc 2007
            # self.img_list.extend(glob.glob(os.path.join(os.path.join(self.data_root, data_), '*/VOC2007/JPEGImages/*.jpg')))
            # self.anno_list.extend(glob.glob(os.path.join(os.path.join(self.data_root, data_), '*/VOC2007/Annotations/*.xml')))

        # for debug
        self.img_list = sorted(self.img_list)
        self.anno_list = sorted(self.anno_list)

        self.class_idx_dict = {class_name: i for i, class_name in enumerate(self.class_names)}     # class name : idx
        self.idx_class_dict = {i: class_name for i, class_name in enumerate(self.class_names)}     # idx : class name

    def _load_image(self, index):
        return Image.open(self.img_list[index]).convert('RGB')

    def _load_anno(self, index):
        return self.anno_list[index]

    def __getitem__(self, idx):

        image = self._load_image(idx)
        anno = self._load_anno(idx)
        boxes, labels = self.parse_voc(anno)
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        w, h = image.size
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.mosaic_transform:
            if random.random() > 0.5:
                # load mosaic img
                image, boxes, labels = load_mosaic(image_id=None,
                                                   len_of_dataset=self.__len__(),
                                                   _load_image=self._load_image,
                                                   _load_anno=self._load_anno,
                                                   _parse=self.parse_voc,
                                                   image=image,
                                                   boxes=boxes,
                                                   labels=labels)

        if self.split == "test":

            info = {}
            img_name = os.path.basename(self.anno_list[idx]).split('.')[0]
            info['name'] = img_name
            info["orig_size"] = torch.as_tensor([int(h), int(w)])

        # --------------------------- for transform ---------------------------
        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.visualization:
            from utils.visualize_boxes import visualize_boxes_labels
            visualize_boxes_labels(image, boxes, labels, data_type='voc', num_labels=21)

        if self.boxes_coord == 'cxywh':
            boxes = xy_to_cxcy(boxes)

        if self.split == "test":
            return image, boxes, labels, info

        return image, boxes, labels

    def __len__(self):
        return len(self.img_list)

    def parse_voc(self, xml_file_path):

        tree = parse(xml_file_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.iter("object"):

            # if int(obj.find('difficult').text) == 1:
            #     # difficult.append(int(obj.find('difficult').text))
            #     continue

            # 'name' tag 에서 멈추기
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            labels.append(self.class_idx_dict[class_name])

            # bbox tag 에서 멈추기
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1

            boxes.append([x_min, y_min, x_max, y_max])

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64))

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = list()
        boxes = list()
        labels = list()
        if self.split == "test":
            info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            if self.split == "test":
                info.append(b[3])

        images = torch.stack(images, dim=0)
        if self.split == "test":
            return images, boxes, labels, info
        return images, boxes, labels


if __name__ == "__main__":
    import datasets.transforms_ as T
    device = torch.device('cuda:0')

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    transform_train = T.Compose([
        T.RandomPhotoDistortion(),
        T.RandomHorizontalFlip(),
        T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
            T.Compose([
                T.RandomResize([400, 500, 600]),
                T.RandomSizeCrop(384, 600),
                T.RandomResize(scales, max_size=1333),
            ]),
        ),
        T.RandomZoomOut(max_scale=2),
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        # FIXME add resize for fixed size image
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = VOC_Dataset(data_root = 'D:\data\\voc',
                            split='train',
                            download=True,
                            transform=transform_train,
                            mosaic_transform=True,
                            visualization=True)

    images, boxes, labels = train_set.__getitem__(0)
    images, boxes, labels = train_set.__getitem__(1)
    images, boxes, labels = train_set.__getitem__(2)
    print(images)
    print(boxes)
    print(labels)

