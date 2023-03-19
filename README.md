# DETR pytorch

re-implementation of detr
Please refer to https://arxiv.org/abs/2005.12872
 
### Properties of this repo.
- [x] Not Use the "Nested Tensor"
- [x] Not Use nn.MultiheadAttention, instead of use timm like transformer(from scratch)
- [x] Change Hungarian mather efficiently
- [x] Able to Much more data augmentation including Mosaic
- [x] Use Voc dataset
- [x] Even though much fixing, the performance is on far the official repo.
- [x] Not distinguish the learning rate between transformer and backbone.
- [x] Fix the resolution of images

### To Do
- [ ] Segmentation
- [ ] voc experiments

### Training Setting
```
- batch size : 36 (official - 64)
- optimizer : Adamw
- epoch : 500
- lr : 1e-4 
- weight decay : 1e-4
- scheduler : step LR (*0.1 at epoch 400)
```

### Results

- quantitative results

|methods        | Traning Dataset        |    Testing Dataset     | Resolution.  | AP               |
|---------------|------------------------| ---------------------- | ------------ | ---------------- |
|papers         | COCOtrain2017          |  COCO val2017(minival) | 800 ~ 1333   | 42.0 (500 epoch) |
|this repo      | COCOtrain2017          |  COCO val2017(minival) | 1024 x 1024  | 41.9 (500 epoch) |

```
Accumulating evaluation results...
DONE (t=6.23s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.419
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.616
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.337
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.538
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
mAP :  0.4190736759071826
mean Loss :  8.931809981664022
Eval Time : 66.7581
```

- qualitative reusults

![results](https://user-images.githubusercontent.com/18729104/221108742-09ded1a8-dcf2-41df-9485-b659e3b6ca08.png)

- attention results

![attention](https://user-images.githubusercontent.com/18729104/223943742-93f7a8d2-4a82-4cf5-92a5-bc4df66a9a72.JPG)

- test as pretrained model
1. download .pth.tar file from 
https://drive.google.com/file/d/1BfgWrkkX2v_d3sbLtIrguZTtIy-MRA5K/view?usp=share_link

2. make ./.logs/detr_coco/saves and put it in .pth.tar file

3. run main for eval
```
python main.py --config ./configs/detr_coco_test.txt
```
