import torch
import torch.nn as nn
from util import box_ops
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (get_world_size, is_dist_avail_and_initialized)


class HungarianLoss(nn.Module):

    def __init__(self, num_classes, weight_dict):

        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict

        # for empty_weight
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = 0.1  # (eos_coef)
        self.register_buffer('empty_weight', empty_weight)

    @torch.no_grad()
    def make_target(self, outputs, boxes, labels):

        # linear_sum_assignment (hungarian)
        batch_size, num_queries = outputs["pred_logits"].shape[:2]  # B, 100
        out_prob_ = outputs["pred_logits"].softmax(-1)  # [B, 100, 92]
        out_bbox_ = outputs["pred_boxes"]  # [B, 100, 4 ]

        device = out_bbox_.device

        # self.num_classes : 91
        target_classes = torch.full([batch_size, num_queries], self.num_classes, dtype=torch.int64, device=device)
        target_boxes = torch.full([batch_size, num_queries, 4], -1, dtype=torch.float32, device=device)

        for b in range(batch_size):
            # the shape of all cost matrix is [num_queries, num_obj] e.g  100, 8
            cost_class = -out_prob_[b][..., labels[b]]
            cost_bbox = torch.cdist(out_bbox_[b], boxes[b], p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_[b]), box_cxcywh_to_xyxy(boxes[b]))
            cost_matrix = 5 * cost_bbox + 1 * cost_class + 2 * cost_giou
            qry_idx, obj_idx = linear_sum_assignment(cost_matrix.cpu())

            # qrt, obj idx 로 permutation 적용한 class, boxes 만듦
            qry_idx = torch.from_numpy(qry_idx).type(torch.int64)
            obj_idx = torch.from_numpy(obj_idx).type(torch.int64)

            # 정답의 위치를 조정해서 sigma 를 적용
            target_classes[b, qry_idx] = labels[b][obj_idx]
            target_boxes[b, qry_idx] = boxes[b][obj_idx]

        return target_boxes, target_classes

    def forward(self, outputs, boxes, labels):

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(l) for l in labels)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        target_boxes, target_classes = self.make_target(outputs, boxes, labels)

        # cls loss
        loss_ce = F.cross_entropy(outputs['pred_logits'].transpose(1, 2), target_classes, self.empty_weight)

        # box loss
        mask = ((target_boxes != -1).sum(-1) / 4).type(torch.bool)
        loss_bbox = F.l1_loss(outputs['pred_boxes'][mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_ce'] = loss_ce
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                outputs = aux_outputs
                target_boxes, target_classes = self.make_target(outputs, boxes, labels)

                # cls loss
                loss_ce = F.cross_entropy(outputs['pred_logits'].transpose(1, 2), target_classes, self.empty_weight)

                # box loss
                mask = ((target_boxes != -1).sum(-1) / 4).type(torch.bool)
                loss_bbox = F.l1_loss(outputs['pred_boxes'][mask], target_boxes[mask], reduction='none')
                loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(outputs['pred_boxes'][mask]),
                    box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

                losses[f'loss_ce_{i}'] = loss_ce
                losses[f'loss_bbox_{i}'] = loss_bbox.sum() / num_boxes
                losses[f'loss_giou_{i}'] = loss_giou.sum() / num_boxes

        return losses