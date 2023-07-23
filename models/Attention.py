import torch
import torch.nn as nn
from math import sqrt
from models.Dlinear import series_decomp
from layers.Embed import PositionalEmbedding
class attn(nn.Module):#Point-Wise-Attention
    def __init__(self, seq_len, pred_len, n_heads=2, attention_dropout=0.0):
        super(attn, self).__init__()
        self.query_projection = nn.Linear(seq_len, pred_len)
        self.key_projection = nn.Linear(seq_len, pred_len)#seq_len)
        self.value_projection = nn.Linear(seq_len, pred_len)#seq_len)
        self.out_projection = nn.Linear(pred_len, pred_len)
        self.dropout = nn.Dropout(attention_dropout)
        self.H = n_heads
        self.seq_len = seq_len
        self.pred_len = pred_len
    def forward(self, x):
        H = self.H
        B = x.shape[0]
        queries = self.query_projection(x).view(B, self.pred_len // H, H)
        keys = self.key_projection(x).view(B, self.pred_len // H, H)
        values = self.value_projection(x).view(B, self.pred_len // H, H)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = self.dropout(torch.softmax(scores / sqrt(H), dim=-1))
        V = torch.einsum("bls,bse->ble", A, values).contiguous()
        V = V.view(B, self.pred_len)
        V = self.out_projection(V)
        return V


class Model(nn.Module):
    def __init__(self, configs):#seq_len, pred_len):
        super(Model, self).__init__()
        self.attn = attn(configs.seq_len, configs.pred_len)
        self.pos = PositionalEmbedding(d_model=configs.seq_len)
    def forward(self, x):
        n_vars = x.shape[-1]
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc

        enc_in = x.transpose(-1, 1)
        enc_in = enc_in + self.pos(enc_in)
        enc_in = torch.reshape(enc_in, (enc_in.shape[0] * enc_in.shape[1], enc_in.shape[2]))
        dec_out = self.attn(enc_in)
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1]))
        dec_out = dec_out.transpose(-1, 1)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out