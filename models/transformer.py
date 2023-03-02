import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pe import get_positional_encoding


class MHA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_k, x_v):
        B, N_q, C = x_q.shape
        _, N_kv, C = x_k.shape
        _, N_kv, C = x_v.shape

        q, k, v = self.q(x_q), self.k(x_k), self.v(x_v)                    # b, n_*, h*d
        _q, _k, _v = map(lambda x: x.reshape(B, -1, self.num_heads, C // self.num_heads), [q, k, v])   # b, n_*, h, d
        qk = torch.einsum('bnhd,bmhd->bhnm', (_q, _k))                     # b, h, n_q, n_k
        att = F.softmax(qk * self.scale, dim=3)                            # b, h, n_q, n_k
        att_out = torch.einsum('bhnm,bmhd->bnhd', (att, _v))               # b, n_q, h, d
        x = att_out.reshape(B, -1, C)                                      # b, n_q, h*d

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, dropout=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = MHA(dim=d_model, attn_drop=dropout, proj_drop=dropout)
        self.mlp = MLP(in_features=d_model, hidden_features=d_feedforward, out_features=d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        # 1. Positional Encoding for q, k
        pe = get_positional_encoding(x)

        # 2. Multi head self attention for encoder
        x = self.norm1(x + self.attn(x + pe, x + pe, x))
        x = self.norm2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers=6, dropout=0.1):
        super().__init__()

        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, d_feedforward=2048, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn_t = MHA(dim=d_model, attn_drop=dropout, proj_drop=dropout)
        self.attn_x = MHA(dim=d_model, attn_drop=dropout, proj_drop=dropout)
        self.mlp = MLP(in_features=d_model, hidden_features=d_feedforward, out_features=d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, t, x, query_embed):
        '''
        :param t: [N, 100, 256] (target)
        :param x: [N, 1024, 256]
        :param query_embed: [N, 100, 256]
        :return:
        '''
        # 1. Query Embedding
        qe = query_embed

        # 2. Positional Encoding for q, k
        pe = get_positional_encoding(x)

        # 3. attn
        t = self.norm1(t + self.attn_t(t + qe, t + qe, t))
        t = self.norm2(t + self.attn_x(t + qe, x + pe, x))
        t = self.norm3(t + self.mlp(t))
        return t


class Decoder(nn.Module):
    def __init__(self, d_model, num_layers=6, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.decoders = nn.ModuleList([
            DecoderLayer(d_model, d_feedforward=2048, dropout=dropout)
            for _ in range(num_layers)])

    def forward(self, t, x, query_embed):
        intermediate = []
        for dec in self.decoders:
            t = dec(t, x, query_embed)
            intermediate.append(self.norm(t))

        return torch.stack(intermediate)  # [6, B, 100, 256]


class Transformer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(d_model, num_layers=6, dropout=dropout)
        self.decoder = Decoder(d_model, num_layers=6, dropout=dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.manual_seed(0)
                nn.init.xavier_uniform_(p)

    def forward(self, x, query_embed):

        bs, c, h, w = x.shape                # [B, 256, 32, 32]
        x = x.flatten(2).permute(0, 2, 1)    # [B, 256, 1024] -> [B, 1024, 256]
        query_embed = query_embed.unsqueeze(0).expand([bs] + list(query_embed.size()))

        # FIXME which condition is better?
        # make target
        # tgt = torch.randn_like(query_embed)
        t = torch.zeros_like(query_embed)

        # encoder, decoder
        x = self.encoder(x)
        t = self.decoder(t, x, query_embed)

        return t, x


if __name__ == '__main__':

    # Transformer
    x = torch.randn([2, 256, 32, 32])
    query_embed = torch.randn([100, 256])
    transformer = Transformer(d_model=256,)
    out = transformer(x, query_embed)
    print(out[0].size())