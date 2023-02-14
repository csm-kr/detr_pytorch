import torch
from models.model import DETR
from torch.nn.parallel import DistributedDataParallel as DDP


def build_model(opts):
    model = DETR(num_classes=opts.num_classes, num_queries=100, d_model=256)
    if opts.distributed:
        model = model.cuda(int(opts.gpu_ids[opts.rank]))
        model = DDP(module=model,
                    device_ids=[int(opts.gpu_ids[opts.rank])],
                    find_unused_parameters=True)
    else:
        # IF DP
        model = torch.nn.DataParallel(module=model, device_ids=[int(id) for id in opts.gpu_ids])
    return model
