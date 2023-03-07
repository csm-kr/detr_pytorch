from torch.utils.data import DataLoader
from datasets.voc_dataset import VOC_Dataset
from datasets.coco_dataset import COCO_Dataset
from torch.utils.data.distributed import DistributedSampler
import datasets.transforms_ as T


def build_dataloader(opts):

    if opts.resize == 0:
        size = 800
        max_size = 1333
    else:
        size = (opts.resize, opts.resize)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # -------------------- transform --------------------
    # FRCNN
    # transform_train = T.Compose([
    #     T.RandomPhotoDistortion(),
    #     T.RandomHorizontalFlip(),
    #     T.Resize(size=size, max_size=max_size),
    #     normalize
    # ])

    # -------------------- transform --------------------
    # DETR
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
        T.Resize(size),
        normalize
    ])

    # -------------------- transform --------------------
    # # DETR + (photo + zoom out + crop)
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # transform_train = T.Compose([
    #     T.RandomHorizontalFlip(),
    #
    #     T.RandomPhotoDistortion(),
    #     T.RandomZoomOut(max_scale=4),
    #     T.RandomCrop(),
    #
    #     T.RandomSelect(
    #         T.RandomResize(scales, max_size=1333),
    #         T.Compose([
    #             T.RandomResize([400, 500, 600]),
    #             T.RandomSizeCrop(384, 600),
    #             T.RandomResize(scales, max_size=1333),
    #         ]),
    #     ),
    #     T.Resize(size),
    #     normalize
    # ])

    # -------------------- transform --------------------
    # DETR + YOLO
    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    # transform_train = T.Compose([
    #     T.RandomHorizontalFlip(),
    #     T.RandomPhotoDistortion(),
    #     T.RandomSelect(
    #         # YOLO
    #         T.Compose([
    #             T.RandomZoomOut(max_scale=4),
    #             T.RandomCrop(),
    #             T.RandomResize(scales, max_size=1333),
    #         ]),
    #         # DETR
    #         T.Compose([
    #             T.RandomSelect(
    #             T.RandomResize(scales, max_size=1333),
    #             T.Compose([
    #                 T.RandomResize([400, 500, 600]),
    #                 T.RandomSizeCrop(384, 600),
    #                 T.RandomResize(scales, max_size=1333),
    #             ]),
    #         ),]),
    #     ),
    #     T.Resize(size),
    #     normalize
    # ])
    # -------------------- transform --------------------

    transform_test = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.Resize(size),
        normalize
    ])

    train_loader = None
    test_loader = None

    if opts.data_type == 'voc':
        train_set = VOC_Dataset(opts.data_root,
                                split='train',
                                download=True,
                                transform=transform_train,
                                mosaic_transform=opts.mosaic_transform,
                                visualization=False)

        test_set = VOC_Dataset(opts.data_root,
                               split='test',
                               download=True,
                               transform=transform_test,
                               visualization=False)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        if opts.distributed:
            train_loader = DataLoader(train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      collate_fn=train_set.collate_fn,
                                      shuffle=False,
                                      num_workers=int(opts.num_workers / opts.world_size),
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset=train_set),
                                      drop_last=False)

        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=0,
                                 pin_memory=True)
        opts.num_classes = 21

    elif opts.data_type == 'coco':

        train_set = COCO_Dataset(data_root=opts.data_root,
                                 split='train',
                                 download=True,
                                 transform=transform_train,
                                 mosaic_transform=opts.mosaic_transform,
                                 boxes_coord=opts.boxes_coord,
                                 visualization=False)

        test_set = COCO_Dataset(data_root=opts.data_root,
                                split='val',
                                download=True,
                                transform=transform_test,
                                boxes_coord=opts.boxes_coord,
                                visualization=False)

        train_loader = DataLoader(train_set,
                                  batch_size=opts.batch_size,
                                  collate_fn=train_set.collate_fn,
                                  shuffle=True,
                                  num_workers=opts.num_workers,
                                  pin_memory=True)

        test_loader = DataLoader(test_set,
                                 batch_size=opts.batch_size,
                                 collate_fn=test_set.collate_fn,
                                 shuffle=False,
                                 num_workers=opts.num_workers,
                                 pin_memory=True)

        if opts.distributed:
            train_loader = DataLoader(train_set,
                                      batch_size=int(opts.batch_size / opts.world_size),
                                      collate_fn=train_set.collate_fn,
                                      shuffle=False,
                                      num_workers=int(opts.num_workers / opts.world_size),
                                      pin_memory=True,
                                      sampler=DistributedSampler(dataset=train_set),
                                      drop_last=True)

            test_loader = DataLoader(test_set,
                                     batch_size=int(opts.batch_size / opts.world_size),
                                     collate_fn=test_set.collate_fn,
                                     shuffle=False,
                                     num_workers=int(opts.num_workers / opts.world_size),
                                     pin_memory=True,
                                     sampler=DistributedSampler(dataset=test_set, shuffle=False),
                                     drop_last=False)

        opts.num_classes = 91

    return train_loader, test_loader



