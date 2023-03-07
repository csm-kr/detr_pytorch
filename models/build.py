from models.model import DETR


def build_model(opts):
    num_classes = opts.num_classes  # 91
    num_queries = opts.num_queries  # 100
    d_model = opts.d_model          # 256
    model = DETR(num_classes, num_queries, d_model).cuda()
    return model
