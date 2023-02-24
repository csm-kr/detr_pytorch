import math
import torch


def get_positional_encoding(tensor, num_pos_feats=128, bld=True):
    temperature = 10000
    scale = math.pi * 2
    num_pos_feats = num_pos_feats
    pos_embed = make_pos(tensor.device, scale, temperature, num_pos_feats)  # [32, 32, 256]
    batch_pos_embed = get_batch_pos_embed(pos_embed, tensor)                # [b, 1024, 256]
    if not bld:
        batch_pos_embed = batch_pos_embed.permute(1, 0, 2)                  # [1024, b, 256]
    return batch_pos_embed


def get_batch_pos_embed(pos_embed, tensor):
    bs = tensor.size(0)
    pos = pos_embed.expand([bs] + list(pos_embed.size()))  # [B, 32, 32, 256]
    pos = pos.reshape([bs, 1024, 256])                       # [B, 1024, 256]

    return pos


def make_pos(device, scale, temperature, num_pos_feats):
    # 29 line and 39 line must update!
    x_ = torch.arange(1, 33, dtype=torch.float32, device=device)
    y_ = torch.arange(1, 33, dtype=torch.float32, device=device)
    y_emb, x_emb = torch.meshgrid(x_, y_, indexing='ij')
    # y_emb, x_emb = torch.meshgrid(x_, y_)

    # normalize:
    eps = 1e-6
    y_emb = y_emb / (y_emb[-1:, :] + eps) * scale  # 가장 큰 값으로 나누고
    x_emb = x_emb / (x_emb[:, -1:] + eps) * scale

    # 0 ~ 128 (pos)
    i = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    div_term = temperature ** (2 * (torch.div(i, 2, rounding_mode='floor')) / num_pos_feats)
    # div_term = temperature ** (2 * (i // 2) / num_pos_feats)

    # 차원늘리기 [32, 32, 1] / [128] -> [32, 32, 128]
    pos_x_ = x_emb[:, :, None] / div_term
    pos_y_ = y_emb[:, :, None] / div_term

    pos_x_ = torch.stack((pos_x_[:, :, 0::2].sin(), pos_x_[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y_ = torch.stack((pos_y_[:, :, 0::2].sin(), pos_y_[:, :, 1::2].cos()), dim=3).flatten(2)

    # [32, 32, 256]
    pos_ = torch.cat((pos_y_, pos_x_), dim=2)

    # [32, 32, 256] -> [256, 32, 32]
    # pos_ = torch.cat((pos_y_, pos_x_), dim=2).permute(2, 0, 1)
    return pos_

# class DETR_PE2d:
#     def __init__(self, num_pos_feats=128):
#         super().__init__()
#         self.temperature = 10000
#         self.scale = math.pi * 2
#         self.num_pos_feats = num_pos_feats
#         self.pos_embed = self.make_pos()
#
#     def make_pos(self):
#
#         x_ = torch.arange(1, 33, dtype=torch.float32, device=x.device)
#         y_ = torch.arange(1, 33, dtype=torch.float32, device=x.device)
#         y_emb, x_emb = torch.meshgrid(x_, y_)
#
#         # normalize:
#         eps = 1e-6
#         y_emb = y_emb / (y_emb[-1:, :] + eps) * self.scale  # 가장 큰 값으로 나누고
#         x_emb = x_emb / (x_emb[:, -1:] + eps) * self.scale
#
#         # 0 ~ 128 (pos)
#         i = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
#         div_term = self.temperature ** (2 * (i // 2) / self.num_pos_feats)
#
#         # 차원늘리기 [32, 32, 1] / [128] -> [32, 32, 128]
#         pos_x_ = x_emb[:, :, None] / div_term
#         pos_y_ = y_emb[:, :, None] / div_term
#
#         pos_x_ = torch.stack((pos_x_[:, :, 0::2].sin(), pos_x_[:, :, 1::2].cos()), dim=3).flatten(2)
#         pos_y_ = torch.stack((pos_y_[:, :, 0::2].sin(), pos_y_[:, :, 1::2].cos()), dim=3).flatten(2)
#
#         # [32, 32, 256] -> [256, 32, 32]
#         pos_ = torch.cat((pos_y_, pos_x_), dim=2).permute(2, 0, 1)
#         return pos_
#
#     def get_pos_embed(self, x):
#         bs = x.size(0)
#         pos = torch.expand(self.pos_embed, [bs] + list(self.pos_embed.size()))
#         return pos


if __name__ == '__main__':
    q = torch.randn([2, 1024, 256])

    PE = get_positional_encoding(q)
    print(PE.size())


