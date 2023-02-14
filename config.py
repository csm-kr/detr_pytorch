import configargparse


def get_args_parser():
    # * config
    parser = configargparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str, default='./.logs', help='config file path')

    # * visualization
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--train_vis_step', type=int, default=100)
    parser.add_argument('--test_vis_step', type=int, default=100)
    parser.add_argument('--is_load_true', dest='is_load', action='store_true')

    # * data
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--resize', type=int, default=1024)
    parser.add_argument('--is_mosaic_transform_true', dest='mosaic_transform', action='store_true')

    # * model
    parser.add_argument('--num_classes', type=int)

    # * loss
    parser.add_argument('--set_cost_class', default=1, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float, help="Relative classification weight of the no-object class")

    # * train
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    parser.add_argument('--start_epoch', type=int, default=0, help='for resume')

    # * test
    parser.add_argument('--test_epoch', type=str, default='best')
    parser.add_argument('--conf_thres', type=float, default=0.05, help='score threshold - 0.05 for test 0.5 for demo')
    parser.add_argument('--top_k', type=int, default=200, help='set top k for after nms')

    # * demo
    parser.add_argument('--demo_epoch', type=str, default='best')
    parser.add_argument('--demo_root', type=str, help='set demo root')                  # TODO
    parser.add_argument('--demo_image_type', type=str)                                  # TODO
    parser.add_argument('--demo_save_true', dest='demo_save', action='store_true')

    # * distributed
    parser.add_argument('--distributed_true', dest='distributed', action='store_true')
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])   # usage : --gpu_ids 0, 1, 2, 3
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int)
    return parser

