import os
import time
import torch
import numpy as np
from tqdm import tqdm
from evaluation.coco_eval import CocoEvaluator


@torch.no_grad()
def test_and_eval(epoch, device, vis, test_loader, model, criterion, postprocessors, xl_log_saver=None, result_best=None, opts=None, is_load=False):

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())  # 'bbox'
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator = CocoEvaluator(test_loader.dataset.coco, iou_types)
    print('Validation of epoch [{}]'.format(epoch))
    model.eval()

    checkpoint = None
    if is_load:
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

        images = data[0]
        boxes = data[1]
        labels = data[2]
        info = data[3]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        info = [{k: v.to(device) for k, v in i.items()} for i in info]

        outputs = model(images)
        loss_dict = criterion(outputs, boxes, labels)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        sum_loss.append(losses.item())

        orig_target_sizes = torch.stack([i["orig_size"] for i in info], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {i['image_id'].item(): output for i, output in zip(info, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        toc = time.time()

        # ---------- print ----------
        if idx % opts.test_vis_step == 0 or idx == len(test_loader) - 1:
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
                               title='test_loss_{}'.format(opts.name),
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