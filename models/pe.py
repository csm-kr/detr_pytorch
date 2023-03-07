import math
import torch


def get_positional_encoding(tensor, num_pos_feats=128, bld=True):
    # temperature = 10000
    # scale = math.pi * 2
    # pos_embed = make_pos(tensor.device, scale, temperature, num_pos_feats)  # [32, 32, 256]
    pos_embed = pos_enc_2d(length=32, d_model=num_pos_feats, device=tensor.device)  # [32, 32, 256]
    batch_pos_embed = get_batch_pos_embed(pos_embed, tensor)                        # [b, 1024, 256]
    if not bld:
        batch_pos_embed = batch_pos_embed.permute(1, 0, 2)                          # [1024, b, 256]
    return batch_pos_embed


def get_batch_pos_embed(pos_embed, tensor):
    bs = tensor.size(0)
    pos = pos_embed.expand([bs] + list(pos_embed.size()))  # [B, 32, 32, 256]
    pos = pos.reshape([bs, 1024, 256])                       # [B, 1024, 256]

    return pos


def pos_enc_2d(length, d_model, device=None):
    if device is None:
        device = torch.device('cpu')

    scale = math.pi * 2
    n = 10000

    x = torch.arange(1, length + 1, dtype=torch.float32, device=device)
    y = torch.arange(1, length + 1, dtype=torch.float32, device=device)
    # y, x = torch.meshgrid(x, y)
    y, x = torch.meshgrid(x, y, indexing='ij')

    # normalize
    eps = 1e-6
    y_norm = y / (length + eps) * scale
    x_norm = x / (length + eps) * scale

    # 0 ~ 128 (pos)
    i = torch.arange(0, d_model, step=2, dtype=torch.float32, device=device)
    div_term = n ** (i / d_model)

    # expand the dim
    y_emb = y_norm[:, :, None] / div_term
    x_emb = x_norm[:, :, None] / div_term

    pos_y = torch.stack((y_emb.sin(), y_emb.cos()), dim=3).flatten(2)
    pos_x = torch.stack((x_emb.sin(), x_emb.cos()), dim=3).flatten(2)

    # [32, 32, 256]
    pos_emb = torch.cat((pos_y, pos_x), dim=2)
    return pos_emb


if __name__ == '__main__':
    q = torch.randn([2, 1024, 256])

    PE = get_positional_encoding(q)
    print(PE.size())


