from models.model import DETR


def build_model(opts):
    model = DETR(num_classes=opts.num_classes, num_queries=100, d_model=256)
    return model
