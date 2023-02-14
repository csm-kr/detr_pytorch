from losses.matcher import build_matcher
from losses.loss import SetCriterion

def build_loss(opts):
    matcher = build_matcher(opts)
    weight_dict = {'loss_ce': 1, 'loss_bbox': opts.bbox_loss_coef}
    weight_dict['loss_giou'] = opts.giou_loss_coef
    # {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}

    aux_weight_dict = {}
    for i in range(5):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes']
    criterion = SetCriterion(opts.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=opts.eos_coef, losses=losses)
    return criterion