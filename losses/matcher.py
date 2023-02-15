import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class  # 1
        self.cost_bbox = cost_bbox    # 5
        self.cost_giou = cost_giou    # 2
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, boxes, labels):

        bs, num_queries = outputs["pred_logits"].shape[:2]      # B, 100
        out_prob_ = outputs["pred_logits"].softmax(-1)          # [B, 100, 92]
        out_bbox_ = outputs["pred_boxes"]                       # [B, 100, 4 ]

        indices = []
        for b in range(bs):
            # the shape of all cost matrix is [num_queries, num_obj] e.g  100, 8
            cost_class = -out_prob_[b][..., labels[b]]
            cost_bbox = torch.cdist(out_bbox_[b], boxes[b], p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox_[b]), box_cxcywh_to_xyxy(boxes[b]))
            cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            indices.append(linear_sum_assignment(cost_matrix.cpu()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
