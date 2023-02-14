import numpy as np
from utils.label_info import coco_color_array, voc_color_array, coco_label_array, voc_label_array, coco_ids_2_labels, coco_color_array_
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

# import platform


def visualize_boxes_labels(image: torch.Tensor,
                           boxes: torch.Tensor,
                           labels: torch.Tensor,
                           data_type: str,
                           num_labels: int,
                           ):
    # assign to cpu
    image.to('cpu')
    boxes.to('cpu')
    labels.to('cpu')

    # mean, std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # c, h, w 2 h, w, c
    image_vis = np.array(image.permute(1, 2, 0), np.float32)
    image_vis *= std
    image_vis += mean
    image_vis = np.clip(image_vis, 0, 1)

    plt.figure('image_w_boxes')
    plt.imshow(image_vis)

    # scale
    h, w = image.size()[1:]
    boxes[:, 0::2] *= w
    boxes[:, 1::2] *= h

    # get label & color arr
    if data_type == 'voc':
        label_arr = voc_label_array
        color_arr = voc_color_array
    elif data_type == 'coco':
        # * 81 label
        label_arr = coco_label_array
        color_arr = coco_color_array
        if num_labels == 91:
            # * 91 label
            label_dict = coco_ids_2_labels
            color_arr = coco_color_array_

    # show boxes
    for i, box in enumerate(boxes):

        # the coord of boxes is x1y1x2y2 (range : 0 ~ 1)
        x1, y1, x2, y2 = box

        if num_labels == 91:
            # * 91 label
            plt.text(x=x1 - 5,
                     y=y1 - 5,
                     s=label_dict[int(labels[i])],
                     bbox=dict(boxstyle='round4',
                               facecolor=color_arr[labels[i]],
                               alpha=0.9))
        else:
            plt.text(x=x1 - 5,
                     y=y1 - 5,
                     s=label_arr[labels[i]],
                     bbox=dict(boxstyle='round4',
                               facecolor=color_arr[labels[i]],
                               alpha=0.9))

        # box
        plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                      width=x2 - x1,
                                      height=y2 - y1,
                                      linewidth=1,
                                      edgecolor=color_arr[labels[i]],
                                      facecolor='none'))

    plt.show()