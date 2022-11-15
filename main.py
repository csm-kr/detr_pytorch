import os
import torch
import visdom
import util.misc as utils
# data
from datasets import build_coco, get_coco_api_from_dataset
# model
from models import build_model
from utils import init_for_distributed
from torch.utils.data import DataLoader, DistributedSampler
from log import XLLogSaver
# train & test
from train import train_one_epoch
from test import test_and_eval


def main_worker(rank, opts):

    # 1. ** argparser **
    print(opts)

    if opts.distributed:
        init_for_distributed(rank, opts)

    # 2. ** device **
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. ** visdom **
    vis = visdom.Visdom(port=opts.visdom_port)
    # vis = None

    # 4. ** dataloader **
    dataset_train = build_coco(image_set='train', args=opts)
    dataset_val = build_coco(image_set='val', args=opts)

    train_loader = None
    test_loader = None

    if opts.distributed:
        # for train
        train_loader = DataLoader(dataset_train,
                                  batch_size=int(opts.batch_size / opts.world_size),
                                  collate_fn=utils.collate_fn,
                                  shuffle=False,
                                  num_workers=int(opts.num_workers / opts.world_size),
                                  pin_memory=True,
                                  sampler=DistributedSampler(dataset=dataset_train),
                                  drop_last=True)

        test_loader = DataLoader(dataset_val,
                                 batch_size=int(opts.batch_size / opts.world_size),
                                 collate_fn=utils.collate_fn,
                                 shuffle=False,
                                 num_workers=int(opts.num_workers / opts.world_size),
                                 pin_memory=True,
                                 sampler=DistributedSampler(dataset=dataset_val, shuffle=False),
                                 drop_last=False)

    else:
        # for eval
        train_loader = DataLoader(dataset_train,
                                  batch_size=opts.batch_size,
                                  collate_fn=utils.collate_fn,
                                  shuffle=False,
                                  num_workers=int(opts.num_workers / opts.world_size),
                                  pin_memory=True,
                                  sampler=torch.utils.data.RandomSampler(dataset_train),
                                  drop_last=True)

        test_loader = DataLoader(dataset_val,
                                 batch_size=opts.batch_size,
                                 collate_fn=utils.collate_fn,
                                 shuffle=False,
                                 num_workers=int(opts.num_workers / opts.world_size),
                                 pin_memory=True,
                                 sampler=torch.utils.data.SequentialSampler(dataset_val),
                                 drop_last=False)

    # 5. ** model **
    model, criterion, postprocessors = build_model(opts)
    model.to(device)

    model_without_ddp = model
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(opts.gpu_ids[opts.rank])])
        model_without_ddp = model.module

    # 6. optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": opts.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts,
                                  lr=opts.lr,
                                  weight_decay=opts.weight_decay)

    base_ds = get_coco_api_from_dataset(dataset_val)

    # 7. logger
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP', 'val_loss'))

    # 8. set best results
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # 9. lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.lr_drop)

    # 10. resume
    if opts.resume:
        if opts.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                opts.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(opts.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not opts.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opts.start_epoch = checkpoint['epoch'] + 1

    if opts.eval:
        test_and_eval(0, base_ds, device, vis, test_loader, model, criterion, postprocessors, opts=opts)
        return

    for epoch in range(opts.start_epoch, opts.epochs):
        if opts.distributed:
            train_loader.sampler.set_epoch(epoch)

        # 11. train
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            opts.clip_max_norm, opts, vis)

        lr_scheduler.step()

        # 12. test
        test_and_eval(
            epoch, base_ds, device, vis, test_loader, model, criterion,
            postprocessors, xl_log_saver, result_best, opts)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    from config import get_args_parser
    import configargparse

    parser = configargparse.ArgumentParser('DETR', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True
    else:
        opts.distributed = False

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)
