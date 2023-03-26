from losses.loss import HungarianLoss


def build_loss(opts):
    weight_dict = {'loss_ce': opts.set_cost_class, 'loss_bbox': opts.set_cost_bbox, 'loss_giou': opts.set_cost_giou}
    aux_weight_dict = {}
    for i in range(opts.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    num_classes = opts.num_classes  # 91
    criterion = HungarianLoss(num_classes, weight_dict=weight_dict)
    return criterion