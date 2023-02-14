import os
import torch
import visdom

# data
from datasets.build import build_dataloader
# model
from models.build import build_model
# loss
from losses.build import build_loss
# log
from log import XLLogSaver
# train & test
from train import train_one_epoch
from test import test_and_eval

# distributed
from utils import init_for_distributed
from models.postprocessor import PostProcess


def main_worker(rank, opts):

    # 1. ** opts **
    print(opts)

    if opts.distributed:
        init_for_distributed(rank, opts)

    # 2. ** device **
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. ** visdom **
    vis = visdom.Visdom(port=opts.visdom_port)

    # 4. ** dataloader **
    train_loader, test_loader = build_dataloader(opts)

    # 5. ** model **
    model = build_model(opts)

    # 6. ** loss **
    criterion = build_loss(opts).to(device)
    postprocessors = {'bbox': PostProcess()}

    model.to(device)
    criterion.to(device)

    # 7. ** optimizer **
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opts.lr,
                                  weight_decay=opts.weight_decay)

    # 8. ** logger **
    xl_log_saver = None
    if opts.rank == 0:
        xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.log_dir, opts.name),
                                  xl_file_name=opts.name,
                                  tabs=('epoch', 'mAP', 'val_loss'))

    # 9. ** set best results **
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # 10. ** lr_scheduler **
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opts.lr_drop)

    for epoch in range(opts.start_epoch, opts.epochs):
        if opts.distributed:
            train_loader.sampler.set_epoch(epoch)

        # 11. ** train **
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch,
            opts.clip_max_norm, opts, vis)

        lr_scheduler.step()

        # 12. ** test **
        test_and_eval(
            epoch, device, vis, test_loader, model, criterion,
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