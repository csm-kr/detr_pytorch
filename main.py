import os
import torch
import visdom

# data
from datasets.build import build_dataloader
# model
from models.build import build_model
from models.postprocessor import PostProcess
# loss
from losses.build import build_loss

from utils import init_for_distributed
from utils.util import resume
from log import XLLogSaver

# train & test
from train import train_one_epoch
from test import test_and_eval


def main_worker(rank, opts):

    # 1. argparser
    print(opts)

    if opts.distributed:
        init_for_distributed(rank, opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    if opts.eval:
        vis = None
    else:
        vis = visdom.Visdom(port=opts.visdom_port)

    # 4. data
    train_loader, test_loader = build_dataloader(opts)

    # 5. model
    model = build_model(opts)
    model.to(device)
    postprocessors = {'bbox': PostProcess()}

    # 6. loss
    criterion = build_loss(opts)
    criterion = criterion.to(device)
    criterion.to(device)

    model_without_ddp = model
    if opts.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[int(opts.gpu_ids[opts.rank])])
        model_without_ddp = model.module

    # 7. optimizer
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

    # 8. logger
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP', 'val_loss'))

    # 9. set best results
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # 10. lr_scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.lr_drop)

    # resume
    if opts.resume:
        model, optimizer, scheduler = resume(opts, model, optimizer, lr_scheduler)

    # only eval
    if opts.eval:
        test_and_eval(opts.test_epoch, device, vis, test_loader, model, criterion, postprocessors,
                      opts=opts, is_load=True)
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
            epoch, device, vis, test_loader, model, criterion,
            postprocessors,
            optimizer, lr_scheduler,
            xl_log_saver, result_best, opts)


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
