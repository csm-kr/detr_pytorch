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
|this repo      | COCOtrain2017          |  COCO val2017(minival) | 1024 x 1024  | 41.1 (300 epoch) |

- qualitative reusults

![results](https://user-images.githubusercontent.com/18729104/221108742-09ded1a8-dcf2-41df-9485-b659e3b6ca08.png)

- attention results

![attention](https://user-images.githubusercontent.com/18729104/223943742-93f7a8d2-4a82-4cf5-92a5-bc4df66a9a72.JPG)
