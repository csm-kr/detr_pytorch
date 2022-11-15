import configargparse


def get_args_parser():
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='config file path')

    parser.add_argument('--name', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='./logs', help='config file path')


    # vis
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--vis_step', type=int, default=100)

    ################ DETR #####################
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='D:\data\coco')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./saves',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    ###########################################

    # parser.set_defaults(distributed=False)
    parser.add_argument('--gpu_ids', nargs="+")
    parser.add_argument('--rank', type=int)

    # # config
    # parser.add_argument('--config', is_config_file=True, help='config file path')
    # parser.add_argument('--name', type=str)
    #
    # # vis
    # parser.add_argument('--visdom_port', type=int, default=8097)
    # parser.add_argument('--vis_step', type=int, default=100)
    #
    # # data
    # parser.add_argument('--data_root', type=str)
    # parser.add_argument('--data_type', type=str)
    # parser.add_argument('--num_classes', type=int)
    #
    # # data augmentation
    # parser.set_defaults(is_vit_data_augmentation=False)  # auto aug, random erasing, label smoothing, cutmix, mixup,
    # parser.add_argument('--mixup_beta', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # parser.add_argument('--vit_data_augmentation_true', dest='is_vit_data_augmentation', action='store_true')
    #
    # # model
    # parser.add_argument('--model_type', type=str)
    # parser.add_argument('--img_size', type=int)
    # parser.add_argument('--patch_size', type=int)
    # parser.add_argument('--in_chans', type=int)
    # parser.add_argument('--embed_dim', type=int)
    # parser.add_argument('--depth', type=int)
    # parser.add_argument('--num_heads', type=int)
    # parser.add_argument('--mlp_ratio', type=float)
    #
    # # qkv_bias false, drop_rate 0.(proj, mlp...) , atte_drop_rate 0. (only attention)
    # parser.add_argument('--drop_path', type=float)
    # parser.set_defaults(has_cls_token=True)
    # parser.add_argument('--has_cls_token_false', dest='has_cls_token', action='store_false')
    # parser.set_defaults(has_last_norm=True)
    # parser.add_argument('--has_last_norm_false', dest='has_last_norm', action='store_false')
    # parser.set_defaults(has_basic_poe=True)
    # parser.add_argument('--has_basic_poe_false', dest='has_basic_poe', action='store_false', help='if not, sinusoid 2d')
    # parser.set_defaults(use_sasa=False)
    # parser.add_argument('--use_sasa_true', dest='use_sasa', action='store_true')
    #
    # parser.set_defaults(has_auto_encoder=False)
    # parser.add_argument('--has_auto_encoder_true', dest='has_auto_encoder', action='store_true')
    # parser.set_defaults(use_gpsa=False)
    # parser.add_argument('--use_gpsa_true', dest='use_gpsa', action='store_true')
    #
    # # train & optimizer
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--epoch', type=int)
    # parser.add_argument('--batch_size', type=int)
    # parser.add_argument('--warmup', type=int, help='warmup epoch')
    # parser.add_argument('--weight_decay', type=float, default=5e-2)
    # parser.add_argument('--save_step', type=int, default=1000, help='if save_step < epoch, then save')
    # parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--log_dir', type=str, default='./logs')
    #
    # # distributed
    # parser.set_defaults(distributed=False)
    # parser.add_argument('--gpu_ids', nargs="+")
    # parser.add_argument('--rank', type=int)
    # parser.add_argument('--world_size', type=int)
    #
    # # test
    # parser.add_argument('--start_epoch', type=int, default=0)
    # parser.add_argument('--test_epoch', type=str, default='best')
    return parser

