import os
from typing import Iterable

import torch

import time
from tqdm import tqdm


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    train_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, opts=None, vis=None):
    # 1. epoch,
    #

    model.train()
    criterion.train()
    tic = time.time()

    for idx, data in enumerate(tqdm(train_loader)):

        images = data[0]
        boxes = data[1]
        labels = data[2]
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        outputs = model(images)
        loss_dict = criterion(outputs, boxes, labels)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        toc = time.time()

        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                lr = param_group['lr']
            elif i == 1:
                backbone_lr = param_group['lr']

        # for each steps
        if (idx % opts.train_vis_step == 0 or idx == len(train_loader) - 1) and opts.rank == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'LR: {lr:.7f} s \t'
                  'Backbone LR: {backbone_lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader),
                          loss=losses,
                          lr=lr,
                          backbone_lr=backbone_lr,
                          time=toc - tic))

            if vis is not None:
            # loss plot
                vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([losses]).unsqueeze(0).cpu(),
                         win='train_loss_' + opts.name,
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='train_loss_{}'.format(opts.name),
                                   legend=['Total Loss']))

    if opts.rank == 0:
        save_path = os.path.join(opts.log_dir, opts.name, 'saves')
        os.makedirs(save_path, exist_ok=True)