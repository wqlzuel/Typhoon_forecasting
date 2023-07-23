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
    def __init__(self, seq_len, pred_len, enc_in=3, dec_in=3, c_out=3, d_model=512, d_ff=512, n_heads=8, activation='gelu', e_layers=2, d_layers=1, dropout=0.0):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.output_attention = False

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Processing
        #self.processing = Dlinear(seq_len, pred_len)
        self.processing = nn.Linear(seq_len, pred_len)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, c_out)
        )
    def forward(self, x_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        #mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        #x_enc = torch.cat([x_enc, mean], dim=1)


        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = self.processing(enc_out.transpose(-1, 1)).transpose(-1, 1)
        dec_out = self.decoder(enc_out)

        #dec_out = self.dec_embedding(x_dec)
        #dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        dec_out = dec_out * std_enc + mean_enc
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]