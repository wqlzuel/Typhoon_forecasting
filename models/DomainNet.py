import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from math import sqrt
from layers.Embed import PositionalEmbedding

class attn(nn.Module):#channel-independent-Attention
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
    def forward(self, x, enc_out):
        H = self.H
        B = x.shape[0]
        queries = self.query_projection(x).view(B, self.pred_len // H, H)
        keys = self.key_projection(enc_out).view(B, self.pred_len // H, H)
        values = self.value_projection(enc_out).view(B, self.pred_len // H, H)
        scores = torch.einsum("ble,bse->bls", queries, keys)
        A = self.dropout(torch.softmax(scores / sqrt(H), dim=-1))
        V = torch.einsum("bls,bse->ble", A, values).contiguous()
        V = V.view(B, self.pred_len)
        V = self.out_projection(V)
        return V

class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):#seq_len, pred_len, enc_in=3, dec_in=3, c_out=3, d_model=512, d_ff=512, n_heads=8, activation='gelu', e_layers=2, d_layers=1, dropout=0.0):
        super(Model, self).__init__()
        self.task = configs.task
        self.pred_len = configs.pred_len
        self.output_attention = False
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),#64
            nn.GELU(),
            nn.Linear(64, 64),  # 6
            nn.GELU(),
            nn.Linear(64, configs.c_out)#3 4 3
        )

        self.decoder = attn(configs.seq_len, 4*configs.pred_len)
        self.decoder2 = attn(4*configs.pred_len, configs.pred_len)
        self.Projection = nn.Linear(4, 1)
        self.pos = PositionalEmbedding(d_model=configs.seq_len)

    '''
    def forward(self, x):
        n_vars = x.shape[-1]
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc

        x_enc = x[:, :, :2].detach().clone()
        x_dec = x.detach().clone()#x.detach().clone()
        enc_out = self.encoder(x_enc).transpose(-1, 1)
        dec_out = x_dec.transpose(-1, 1)
        #dec_out = torch.reshape(dec_out, (dec_out.shape[0] * dec_out.shape[1], dec_out.shape[2]))
        enc_out = torch.reshape(enc_out, (enc_out.shape[0] * enc_out.shape[1], enc_out.shape[2]))
        dec_out = self.decoder(enc_out, dec_out)#dec_out, enc_out)
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1]))

        dec_out = dec_out.transpose(-1, 1)
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
    '''
    def forecast(self, x):
        x = x[:, :, :-1]
        n_vars = x.shape[-1]
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc

        x_enc = x[:, :, :2].detach().clone()
        x_dec = x.detach().clone()  # x.detach().clone()
        enc_out = self.encoder(x_enc).transpose(-1, 1)
        dec_out = x_dec.transpose(-1, 1) + self.pos(x_dec.transpose(-1, 1))
        #dec_out = torch.reshape(dec_out, (dec_out.shape[0] * dec_out.shape[1], dec_out.shape[2]))
        enc_out = torch.reshape(enc_out, (enc_out.shape[0] * enc_out.shape[1], enc_out.shape[2]))
        dec_out = self.decoder(enc_out, dec_out)# dec_out, enc_out)
        dec_out = self.decoder2(dec_out, dec_out)
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1]))

        dec_out = dec_out.transpose(-1, 1)
        dec_out = dec_out * std_enc + mean_enc

        return dec_out[:,:,:-1]
    def classification(self, x):
        n_vars = 4
        #mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        #x = x - mean_enc
        #std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        #x = x / std_enc

        x_enc = x[:, :, :2].detach().clone()
        x_dec = x.detach().clone()
        enc_out = self.encoder(x_enc).transpose(-1, 1)
        dec_out = x_dec.transpose(-1, 1)

        enc_out = torch.reshape(enc_out, (enc_out.shape[0] * enc_out.shape[1], enc_out.shape[2]))
        dec_out = self.decoder(enc_out, dec_out)  # dec_out, enc_out)
        dec_out = torch.reshape(dec_out, (-1, n_vars, dec_out.shape[-1]))

        dec_out = dec_out.transpose(-1, 1)
        dec_out = self.Projection(dec_out).squeeze(-1)# * std_enc + mean_enc  # [:, :, :-1]

        #dec_out = torch.softmax(dec_out, dim=-1).transpose(-1, 1)
        return dec_out#.unsqueeze(-1)
    def forward(self, x, task='f'):#f为预测，c为分类
        task = self.task
        if task=='f':
            dec_out = self.forecast(x)
        else:
            dec_out = self.classification(x)
        return dec_out#[:, -self.pred_len:, :]
