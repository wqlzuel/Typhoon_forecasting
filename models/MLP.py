import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in=3, d_model=64):
        super(Model, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(seq_len*enc_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len)
        )
        #self.pred_len = pred_len
    def forward(self, x):
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc
        x = x.view(x.shape[0], -1)

        dec_out = self.MLP(x)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out