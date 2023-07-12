import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
from math import sqrt
def preprocess(dataframe, min_len=10):
    data=[]
    series=[]
    for i in dataframe.values:
        if '66666' not in i[0].split():
            s=i[0].split()
            d=[int(s[3])]+[int(s[4])]
            series.append(d)
        else:
            if len(series)>(min_len-1):#len(series)>0
                data.append(series)
                series=[]
    return data
class DomainDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # set data
        self.Y = Y  # set lables

    def __len__(self):
        return len(self.X)  # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]] 
read_data = pd.read_csv(r'E:\hah时序数据\bst_all.txt',sep = '\n',header=None)
data = preprocess(read_data)
domain_len = 1000#len(data)#域个数
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
    def __init__(self, seq_len, pred_len):
        super(Model, self).__init__()
        #1,7,2 -> 1,3,2
        #self.emb = nn.Linear(seq_len*2, 2)
        self.s_MLP = nn.Sequential(
                    nn.Linear(seq_len,512),
                    nn.GELU(),#GELU ReLU
                    nn.Linear(512,512),
                    nn.GELU(),#GELU ReLU
                    nn.Linear(512,pred_len)
                    )
        self.t_MLP = nn.Linear(seq_len, pred_len)
        # Decomp
        kernel_size = 25
        self.decomp = series_decomp(kernel_size)
    def forward(self, x):
        mean_enc = x.mean(1, keepdim=True).detach() # B x 1 x E
        x = x - mean_enc
        std_enc = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() # B x 1 x E
        x = x / std_enc
          
        seasonal_init, trend_init = self.decomp(x)
        sea_out = self.s_MLP(seasonal_init.transpose(-1,1))
        tre_out = self.t_MLP(trend_init.transpose(-1,1))
        dec_out = (sea_out+tre_out).transpose(-1,1)

        dec_out = dec_out * std_enc + mean_enc
        return dec_out

model=Model(seq_len=6, pred_len=4)
criterion = nn.L1Loss()
model_optim = optim.Adam(model.parameters(), lr=1e-2)

model.train()
for domain in range(domain_len):    
    domain_train = torch.tensor(data[domain]).float()
    domain_in = domain_train.unfold(0,10,1).transpose(-1,1)#10为seq_len+pred_len
    value = domain_in[:,:6,:]
    label = domain_in[:,6:,:]
    domain_dataset = DomainDataset(value, label)
    dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=1, shuffle=False, drop_last=True)
    for epoch in range(10):
        for batch in dataloader:
            x=batch[0]
            y=batch[1]
            outputs=model(x)
            loss = criterion(outputs, y)
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

test_domain_len = 100
avg=[]
for domain in range(test_domain_len):
    domain_test = torch.tensor(data[domain_len+1+domain]).float()
    test_in = domain_test.unfold(0,10,1).transpose(-1,1)#10为seq_len+pred_len
    test_value = test_in[:,:6,:]
    test_label = test_in[:,6:,:]
    test_dataset = DomainDataset(test_value, test_label)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                            shuffle=False, drop_last=True)
    model.eval()
    avg_loss=[]
    for batch in testloader:
        x=batch[0]
        y=batch[1]
        outputs=model(x)
        loss = criterion(outputs, y)
        avg_loss.append(loss.detach().numpy())
    avg_loss=np.array(avg_loss).mean()
    avg.append(avg_loss)
avg = np.array(avg).mean()
print(avg)
