train(config)
```
--config ./configs/coco/detr_coco_train.txt
```

```
% python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path D:\data\coco
python main.py --batch_size 2  --coco_path D:\data\coco
```

```
Namespace(aux_loss=True, backbone='resnet50', batch_size=2, bbox_loss_coef=5, clip_max_norm=0.1, 
coco_panoptic_path=None, 
coco_path='D:\\data\\coco', dataset_file='coco', dec_layers=6, device='cuda', dice_loss_coef=1, 
dilation=False, dim_feedforward=2048, dist_url='env://', dropout=0.1, enc_layers=6, eos_coef=0.1, 
epochs=300, eval=False, frozen_weights=None, giou_loss_coef=2, hidden_dim=256, lr=0.0001, 
lr_backbone=1e-05, lr_drop=200, mask_loss_coef=1, masks=False, nheads=8, num_queries=100, 
num_workers=2, output_dir='', position_embedding='sine', pre_norm=False, remove_difficult=False, 
resume='', seed=42, set_cost_bbox=5, set_cost_class=1, set_cost_giou=2, start_epoch=0, 
weight_decay=0.0001, world_size=1)
```

evaluation
```
python main.py --batch_size 1 --no_aux_loss --eval --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path D:\data\coco
```

##### train

```
python main.py --config ./configs/coco/detr_coco_train.txt
```

##### test and evaluation

```
python main.py --config ./configs/coco/detr_coco_test.txt
```