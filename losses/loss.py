import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.matcher import build_matcher
from util.misc import (get_world_size, is_dist_avail_and_initialized)
from util import box_ops


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, boxes, labels, indices, num_boxes, log=True):
        ########################################################
        target_classes_ = self.make_target(indices, labels)
        loss_ce_ = F.cross_entropy(outputs['pred_logits'].transpose(1, 2), target_classes_, self.empty_weight)
        losses = {'loss_ce': loss_ce_}
        ########################################################
        return losses

    def loss_boxes(self, outputs, boxes, labels, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        ########################################################
        target_boxes_ = self.make_target(indices, boxes)
        mask = ((target_boxes_ != -1).sum(-1)/4).type(torch.bool)
        losses = {}
        loss_bbox_ = F.l1_loss(outputs['pred_boxes'][mask], target_boxes_[mask], reduction='none')
        loss_giou_ = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes_[mask])))

        losses['loss_bbox'] = loss_bbox_.sum() / num_boxes
        losses['loss_giou'] = loss_giou_.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, boxes, labels, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, boxes, labels, indices, num_boxes, **kwargs)

    def make_target(self, indices, gt):
        bs = len(gt)
        device = gt[0].device

        if gt[0].dim() == 1:
            type = 'labels'
        elif gt[0].dim() == 2:
            type = 'boxes'

        batch_idx = []
        qry_idx = []
        target_with_perm = []

        for i, (g, (qry, obj)) in enumerate(zip(gt, indices)):
            batch_idx.append(torch.full_like(qry, i))
            qry_idx.append(qry)
            target_with_perm.append(g[obj])

        batch_idx = torch.cat(batch_idx)
        qry_idx = torch.cat(qry_idx)
        target_with_perm = torch.cat(target_with_perm)

        if type == 'labels':
            target_classes = torch.full([bs, 100], self.num_classes, dtype=torch.int64, device=device)
            target_classes[batch_idx, qry_idx] = target_with_perm
            ret = target_classes
        else:
            target_boxes = torch.full([bs, 100, 4], -1, dtype=torch.float32, device=device)
            target_boxes[batch_idx, qry_idx] = target_with_perm
            ret = target_boxes
        return ret

    def forward(self, outputs, boxes, labels):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, boxes, labels)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(l) for l in labels)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, boxes, labels, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, boxes, labels)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, boxes, labels, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def build_loss(args):
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    aux_weight_dict = {}
    for i in range(args.dec_layers - 1):
        aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    num_classes = args.num_classes # 91

    losses = ['labels', 'boxes']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    return criterion


# if __name__ == '__main__':
