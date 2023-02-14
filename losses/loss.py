import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.box_ops as box_ops
from utils.util import get_world_size, is_dist_avail_and_initialized


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
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        target_classes = self.make_target(indices, labels)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, boxes, labels, indices, num_boxes):
        return 0

    def loss_boxes(self, outputs, boxes, labels, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs

        src_boxes = outputs['pred_boxes']
        target_boxes = self.make_target(indices, boxes)
        mask = ((target_boxes != -1).sum(-1)/4).type(torch.bool)
        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, boxes, labels, indices, num_boxes):
        return 0

    def get_loss(self, loss, outputs, boxes, labels, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
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
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
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
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, boxes, labels, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses




