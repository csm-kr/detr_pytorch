import numpy as np
from utils.label_info import coco_color_array, voc_color_array, coco_label_array, voc_label_array, coco_ids_2_labels, coco_color_array_
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import cv2
import os
# import platform

def visualize_dataset(image: torch.Tensor,
                      boxes: torch.Tensor,
                      labels: torch.Tensor,
                      
                      lib_type: str,
                      data_type: str,
                      num_labels: int,
                      
                      mean = np.array([0.485, 0.456, 0.406]),
                      std = np.array([0.229, 0.224, 0.225])
                      ):
    """
    transform 이후의 dataset을 출력하는 함수
    image : torch.Tensor image range - [0, 1], float32
    boxes : torch.Tensor, [num_obj, 4] - range [0, 1], float32
    label : np.array, [num_obj] int32
    score : np.array, [num_obj] float32

    lib_type : str (cv2, plt) 
    data_type : (coco, voc)
    vis_type : (save, view) 


    """
    # 1. check params
    # assert 이후의 조건이 true 일 때만 그대로 진행. 
    assert lib_type in ['cv2', 'plt'], 'cv2 혹은 plt 의 lib 를 사용해야 합니다.'
    assert data_type in ['coco', 'voc'], 'coco 혹은 voc 의 data 를 사용해야 합니다.'

    # 2. assign to cpu
    image = image.to('cpu')
    boxes = boxes.to('cpu')
    labels = labels.to('cpu')

    # 3. convert [c, h, w] to [h, w, c] and unnormalize (mean, std)
    image_vis = np.array(image.permute(1, 2, 0), np.float32)
    image_vis *= std
    image_vis += mean
    image_vis = np.clip(image_vis, 0, 1)

    # 4. set boxes coordinates
    h, w = image.size()[1:]
    boxes[:, 0::2] *= w
    boxes[:, 1::2] *= h

    if lib_type == 'cv2':
        show_det_cv2(image_vis, boxes, labels)

    return 


def show_det_cv2(image_vis, boxes, labels, ):
    """
    img : np.ndarray, float32
    """

    # for cv2 convert color rgb2bgr
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    # set label and color array
    # if data_type == 'voc':
    #     label_arr = voc_label_array
    #     color_arr = voc_color_array

    # elif data_type == 'coco':
    #     label_arr = coco_label_array
    #     color_arr = coco_color_array

    #     if num_labels == 91:
    #         # * 91 label
    #         label_dict = coco_ids_2_labels
    #         color_arr = coco_color_array_

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = [int(b) for b in box]

        # if num_labels == 91:
        #     text = f'{label_dict[int(labels[i])]}'
        #     if scores is not None:
        #         text = f'{label_dict[int(labels[i])]}: {scores[i]:0.2f}'

        cv2.rectangle(image_vis,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=[0, 0, 0],
                      thickness=2)

        # text_size
        text_size = cv2.getTextSize(text='detection',
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    thickness=1)[0]

        # text_rec
        cv2.rectangle(image_vis,
                      pt1=(x1, y1),
                      pt2=(x1 + text_size[0] + 3, y1 + text_size[1] + 4),
                      color=[0, 0, 0],
                      thickness=-1)

        # put text
        cv2.putText(image_vis,
                    text='detection',
                    org=(x1 + 10, y1 + 10),  # must be int
                    fontFace=0,
                    fontScale=0.4,
                    color=(0, 0, 0))

    cv2.imshow('result', image_vis)
    cv2.waitKey(0)
    
    # os.makedirs('./vis_training', exist_ok=True)

    # cv2 imwrite 는 0~255 int 로 변경 되어서 저장
    # imwrite always expects [0,255]
    # (https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite)
    # cv2.imwrite('./vis_training/a.jpg', image_vis * 255)

    return


def show_det_plt():
    return
    


# TO DO  
# def visualize_demo 함수 만들기 

def visualize_cv2(image: torch.Tensor,
                  boxes: torch.Tensor,
                  labels: torch.Tensor,

                  data_type: str,
                  num_labels: int,

                  scores: torch.Tensor,

                  original_w: int = None,
                  original_h: int = None,
                  name: str = None
                  ) -> None:

    """
    image : torch.Tensor image range - [0, 1], float32
    boxes : np.array, [num_obj, 4] - range [0, 1], float32
    label : np.array, [num_obj] int32
    score : np.array, [num_obj] float32

    """

    # assign to cpu
    image = image.to('cpu')
    boxes = boxes.to('cpu')
    labels = labels.to('cpu')

    # mean, std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # convert [c, h, w] to [h, w, c]
    image_vis = np.array(image.permute(1, 2, 0), np.float32)
    image_vis *= std
    image_vis += mean
    image_vis = np.clip(image_vis, 0, 1)
    # h, w c

    # the image_vis.shape is h, w, c
    image_vis = cv2.resize(image_vis, (original_w, original_h))

    h, w = original_h, original_w
    boxes[:, 0::2] *= w
    boxes[:, 1::2] *= h

    # for cv2 convert color rgb2bgr
    image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

    # set label and color array
    if data_type == 'voc':
        label_arr = voc_label_array
        color_arr = voc_color_array

    elif data_type == 'coco':
        label_arr = coco_label_array
        color_arr = coco_color_array

        if num_labels == 91:
            # * 91 label
            label_dict = coco_ids_2_labels
            color_arr = coco_color_array_

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = [int(b) for b in box]

        if num_labels == 91:
            text = f'{label_dict[int(labels[i])]}'
            if scores is not None:
                text = f'{label_dict[int(labels[i])]}: {scores[i]:0.2f}'

        cv2.rectangle(image_vis,
                      pt1=(x1, y1),
                      pt2=(x2, y2),
                      color=color_arr[labels[i]],
                      thickness=2)

        # text_size
        text_size = cv2.getTextSize(text=text,
                                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                                    fontScale=1,
                                    thickness=1)[0]

        # text_rec
        cv2.rectangle(image_vis,
                      pt1=(x1, y1),
                      pt2=(x1 + text_size[0] + 3, y1 + text_size[1] + 4),
                      color=color_arr[labels[i]],
                      thickness=-1)

        # put text
        cv2.putText(image_vis,
                    text=text,
                    org=(x1 + 10, y1 + 10),  # must be int
                    fontFace=0,
                    fontScale=0.4,
                    color=(0, 0, 0))

    # cv2.imshow('result', image_vis)
    os.makedirs('./demo_results', exist_ok=True)

    # cv2 imwrite 는 0~255 int 로 변경 되어서 저장
    # imwrite always expects [0,255]
    # (https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite)
    cv2.imwrite(f'./demo_results/float_cv2_detection_results_of_{name}', image_vis * 255)
    # cv2.waitKey(0)
    return


def visualize_boxes_labels(image: torch.Tensor,
                           boxes: torch.Tensor,
                           labels: torch.Tensor,

                           data_type: str,
                           num_labels: int,

                           scores: torch.Tensor = None,
                           original_w=None,
                           original_h=None,
                           name=None
                           ):
    # 이 함수의 목적은, 

    # assign to cpu
    image = image.to('cpu')
    boxes = boxes.to('cpu')
    labels = labels.to('cpu')

    # mean, std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # c, h, w 2 h, w, c
    image_vis = np.array(image.permute(1, 2, 0), np.float32)
    image_vis *= std
    image_vis += mean
    image_vis = np.clip(image_vis, 0, 1)
    print("image_vis", image_vis)
    # resize
    # image_vis = cv2.resize(image_vis, (original_w, original_h))

    plt.figure('image_w_boxes')
    plt.imshow(image_vis)

    # scale
    if original_w is None and original_h is None:
        h, w = image.size()[1:]
    h, w = original_h, original_w
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
            text = f'{label_dict[int(labels[i])]}'
            if scores is not None:
                text = f'{label_dict[int(labels[i])]}: {scores[i]:0.2f}'

            # * 91 label
            plt.text(x=x1,
                     y=y1,
                     s=text,
                     fontsize=15,
                     bbox=dict(facecolor=color_arr[labels[i]],
                               alpha=0.5))
        else:
            text = f'{label_arr[labels[i]]}'
            if scores is not None:
                text = f'{label_arr[labels[i]]}: {scores[i]:0.2f}'

            plt.text(x=x1,
                     y=y1,
                     s=text,
                     fontsize=15,
                     bbox=dict(facecolor=color_arr[labels[i]],
                               alpha=0.5))

        # box
        plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                      width=x2 - x1,
                                      height=y2 - y1,
                                      linewidth=2,
                                      edgecolor=color_arr[labels[i]],
                                      facecolor='none'))
    plt.axis('off')
    # plt.savefig - name, facecolor
    if name is not None:
        print('save')
        os.makedirs('./demo_results', exist_ok=True)
        plt.savefig(f'./demo_results/detection_results_of_{name}')
    plt.show()