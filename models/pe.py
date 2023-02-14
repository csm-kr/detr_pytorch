import math
import torch


def get_positional_encoding(tensor, num_pos_feats=128):
    temperature = 10000
    scale = math.pi * 2
    num_pos_feats = num_pos_feats
    pos_embed = make_pos(tensor, scale, temperature, num_pos_feats)
    batch_pos_embed = get_batch_pos_embed(pos_embed, tensor)
    return batch_pos_embed


def get_batch_pos_embed(pos_embed, tensor):
    bs = tensor.size(0)
    pos = pos_embed.expand([bs] + list(pos_embed.size()))  # [B, 32, 32, 256]
    pos = pos.reshape(tensor.size())                       # [B, 1024, 256]
    return pos


def make_pos(x, scale, temperature, num_pos_feats):

    x = torch.arange(1, 33, dtype=torch.float32, device=x.device)
    y = torch.arange(1, 33, dtype=torch.float32, device=x.device)
    y_emb, x_emb = torch.meshgrid(x, y)

    # normalize:
    eps = 1e-6
    y_emb = y_emb / (y_emb[-1:, :] + eps) * scale  # 가장 큰 값으로 나누고
    x_emb = x_emb / (x_emb[:, -1:] + eps) * scale

    # 0 ~ 128 (pos)
    i = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    div_term = temperature ** (2 * (torch.div(i, 2, rounding_mode='floor')) / num_pos_feats)

    # 차원늘리기 [32, 32, 1] / [128] -> [32, 32, 128]
    pos_x = x_emb[:, :, None] / div_term
    pos_y = y_emb[:, :, None] / div_term

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    # [32, 32, 256]
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


if __name__ == '__main__':
    q = torch.randn([2, 1024, 256])

    PE = get_positional_encoding(q)
    print(PE.size())


