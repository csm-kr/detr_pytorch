import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50 # , ResNet50_Weights
from models.transformer import Transformer
import torch.nn.functional as F


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class BoxLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_layers = nn.Sequential(nn.Linear(in_features=256, out_features=256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=256, out_features=256),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(in_features=256, out_features=4),
                                        )

    def forward(self, x):
        x = self.box_layers(x)
        return x


class DETR(nn.Module):
    def __init__(self, num_classes, num_queries, d_model):
        super().__init__()

        # model
        # self.backbone = nn.Sequential(*list(nn.ModuleList(resnet50(weights=ResNet50_Weights.IMAGENET1K_V2,
        #                                                            norm_layer=FrozenBatchNorm2d).children()))[:-2])
        self.backbone = nn.Sequential(*list(nn.ModuleList(resnet50(pretrained=True,
                                                                   norm_layer=FrozenBatchNorm2d).children()))[:-2])

        # resnet의 self.conv1(x), self.bn1(x), self.relu(x), self.maxpool(x), self.layer1(x) 는 학습시키지 않는다.
        for name, parameter in self.backbone.named_parameters():
            if '5.' not in name and '6.' not in name and '7.' not in name:
                parameter.requires_grad_(False)

        self.transformer = Transformer(d_model)
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        self.class_layer = nn.Linear(d_model, num_classes + 1)
        self.box_layer = MLP(256, 256, 4, 3)

        # embedding
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.aux_loss = True
        print("num_params : ", self.count_parameters())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # def forward(self, x):
    def forward(self, samples):
        # mask 없이 tensor 만  출력해서 Nested Tensor 를 사용하지 않음
        x = samples

        x = self.backbone(x)                                       # [B, 2048, 32, 32]
        x = self.input_proj(x)                                     # [B, 256, 32, 32]
        x, _ = self.transformer(x, self.query_embed.weight)

        outputs_class = pred_classes = self.class_layer(x)         # [6, 9, 100, 92]
        outputs_coord = pred_boxes = self.box_layer(x).sigmoid()   # [6, 9, 100, 4]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


if __name__ == '__main__':
    img = torch.randn(2, 3, 1024, 1024).cuda()
    detr = DETR(num_classes=91, num_queries=100, d_model=256).cuda()
    out = detr(img)
    print('decoder layer 6')
    print(out['pred_logits'].size())
    print(out['pred_boxes'].size())
    for i, out_dec in enumerate(out['aux_outputs']):
        print('decoder layer {}'.format(i + 1))
        print(out_dec['pred_logits'].size())
        print(out_dec['pred_boxes'].size())

    '''
    ...
    decoder layer 6
    torch.Size([2, 100, 92])
    torch.Size([2, 100, 4])
    '''
