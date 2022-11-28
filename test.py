import os
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets.coco_eval import CocoEvaluator


@torch.no_grad()
def test_and_eval(epoch, base_ds, device, vis, test_loader, model, criterion, postprocessors, xl_log_saver=None, result_best=None, opts=None, is_load=False):

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())  # 'bbox'
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()

    checkpoint = None
    if opts.is_load:
        f = os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.{}.pth.tar'.format(epoch))
        device = torch.device('cuda:{}'.format(opts.gpu_ids[opts.rank]))
        if isinstance(model, (torch.nn.parallel.distributed.DistributedDataParallel, torch.nn.DataParallel)):
            checkpoint = torch.load(f=f,
                                    map_location=device)
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict)
        else:
            checkpoint = torch.load(f=f,
                                    map_location=device)
            state_dict = checkpoint['model_state_dict']
            state_dict = {k.replace('module.', ''): v for (k, v) in state_dict.items()}
            model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = []

    for idx, data in enumerate(tqdm(test_loader)):

        samples, targets = data

        # ---------- cuda ----------
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        sum_loss.append(losses.item())

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        toc = time.time()

        # ---------- print ----------
        if idx % opts.vis_step == 0 or idx == len(test_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Time : {time:.4f}\t'
                  .format(epoch,
                          idx, len(test_loader),
                          loss=losses,
                          time=toc - tic))

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
    mAP = stats[0]
    mean_loss = np.array(sum_loss).mean()

    print("mAP : ", mAP)
    print("mean Loss : ", mean_loss)
    print("Eval Time : {:.4f}".format(time.time() - tic))

    if opts.rank == 0:
        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss_' + opts.name,
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss and map for {}'.format(opts.name),
                               legend=['test Loss', 'mAP']))

        if xl_log_saver is not None:
            xl_log_saver.insert_each_epoch(contents=(epoch, mAP, mean_loss))

        # save best.pth.tar
        if result_best is not None:
            if result_best['mAP'] < mAP:
                print("update best model")
                result_best['epoch'] = epoch
                result_best['mAP'] = mAP
                if checkpoint is None:
                    checkpoint = {'epoch': epoch,
                                  'model_state_dict': model.state_dict()}
                    torch.save(checkpoint, os.path.join(opts.log_dir, opts.name, 'saves', opts.name + '.best.pth.tar'))