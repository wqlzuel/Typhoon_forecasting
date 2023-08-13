import torch
import torch.nn as nn
from models.LORA import LoraLinear
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 1,7,2 -> 1,3,2
        # self.emb = nn.Linear(seq_len*2, 2)
        self.s_MLP = nn.Linear(configs.seq_len, configs.pred_len)#LoraLinear(seq_len, pred_len, r=8)

        '''
            nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.GELU(),  # GELU ReLU
            nn.Linear(512, 512),
            nn.GELU(),  # GELU ReLU
            nn.Linear(512, pred_len)
        )'''

        self.t_MLP = nn.Linear(configs.seq_len, configs.pred_len)#LoraLinear(seq_len, pred_len, r=8)#nn.Linear(seq_len, pred_len)
        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)

    def forward(self, x):
        x = x[:,:,:-1]
        mean_enc = x.mean(1, keepdim=True).detach()  # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x = x / std_enc


        seasonal_init, trend_init = self.decomp(x)
        sea_out = self.s_MLP(seasonal_init.transpose(-1, 1))
        tre_out = self.t_MLP(trend_init.transpose(-1, 1))
        dec_out = (sea_out + tre_out).transpose(-1, 1)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out[:,:,:-1]

class Dlinear(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(Dlinear, self).__init__()
        self.s_MLP = nn.Linear(seq_len, pred_len)
        self.t_MLP = nn.Linear(seq_len, pred_len)#nn.Linear(seq_len, pred_len)
        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)
    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        sea_out = self.s_MLP(seasonal_init.transpose(-1, 1))
        tre_out = self.t_MLP(trend_init.transpose(-1, 1))
        dec_out = (sea_out + tre_out).transpose(-1, 1)
        return dec_out