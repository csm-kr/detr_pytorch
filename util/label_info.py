import numpy as np

# for voc label
voc_label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
np.random.seed(1)
voc_color_array = np.random.randint(256, size=(21, 3)) / 255
# In plt, rgb color space's range from 0 to 1

# for voc label
coco_label_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                   'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                   'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                   'wine glass', 'cup', 'fork', 'knife', 'spoon',
                   'bowl', 'banana', 'apple', 'sandwich', 'orange',
                   'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                   'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                   'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                   'toaster', 'sink', 'refrigerator', 'book', 'clock',
                   'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
np.random.seed(1)
coco_color_array = np.random.randint(256, size=(81, 3)) / 255
# In plt, rgb color space's range from 0 to 1