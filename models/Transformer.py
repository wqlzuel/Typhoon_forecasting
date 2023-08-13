import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from models.Dlinear import Dlinear
class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):#seq_len, pred_len, enc_in=3, dec_in=3, c_out=3, d_model=512, d_ff=512, n_heads=8, activation='gelu', e_layers=2, d_layers=1, dropout=0.0):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = False
        #configs.d_model = 32
        # Embedding
        #self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        #self.enc_embedding = nn.Linear(configs.enc_in, 1)
        self.fc = nn.Linear(6, 16)#3)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation='gelu'
                ) for l in range(2)#e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation='gelu',
                )
                for l in range(1)#configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
    def forward(self, x):
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc

        x_enc = x[:, :, :2].detach().clone()
        x_dec = x.detach().clone()


        #enc_out = self.enc_embedding(x_enc).transpose(-1, 1)
        enc_out = x_enc.repeat(1, 1, 3)
        #print(enc_out.shape)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.fc(enc_out).transpose(-1, 1)
        dec_out = x_dec.transpose(-1, 1)#self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out.transpose(-1, 1)
        dec_out = dec_out * std_enc + mean_enc

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]