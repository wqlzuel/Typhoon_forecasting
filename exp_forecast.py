from models import Dlinear, Attention, Transformer, DomainNet
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from exp_basic import Exp_Basic, DomainDataset

class Exp_Main_F(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_F, self).__init__(args)
        self.args = args
    def _build_model(self):
        model_dict = {
            'Dlinear': Dlinear,
            'Attention': Attention,
            'Transformer': Transformer,
            'DomainNet': DomainNet
        }
        model = model_dict[self.args.model].Model(self.args).float()
        return model#.to('cuda')

    def preprocess(self, dataframe, min_len=10):  # 16 10
        data = []
        series = []
        for i in dataframe.values:
            if '66666' not in i[0].split():
                s = i[0].split()
                d = [int(s[3])/10] + [int(s[4])/10] + [int(s[5])] + [int(s[2])]
                series.append(d)
            else:
                if len(series) > (min_len - 1):  # len(series)>0
                    data.append(series)
                    series = []
        return data

    def _get_data(self):
        read_data = pd.read_csv('/typhoon/data/bst_all.txt', sep='\t', header=None)
        data = self.preprocess(read_data, min_len=self.args.seq_len+self.args.pred_len)
        return data

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def train(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))  # 300##1000  # len(data)#域个数
        criterion = self._select_criterion()
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        k = 0
        self.model.train()
        domain_train = torch.tensor(data[0])#0
        domain_in = domain_train.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
        value = domain_in[:, :self.args.seq_len, :]
        label = domain_in[:, self.args.seq_len:, :-2]
        for domain in range(domain_len-1):
            domain_train = torch.tensor(data[domain+1]).float()#+1
            domain_in = domain_train.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)  # 10为seq_len+pred_len
            value = torch.concat([value, domain_in[:, :self.args.seq_len, :]], dim=0)
            label = torch.concat([label, domain_in[:, self.args.seq_len:, :-2]], dim=0)
        domain_dataset = DomainDataset(value, label)
        dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=32, shuffle=True, drop_last=False)#32
        train_epoch = self.args.train_epochs  # 200
        #i=0
        for epoch in range(train_epoch):  # 20 10 30
            #i = i+1
            #if i > 40:
                #i=0
            #domain_train = torch.tensor(data[i]).float()
            #domain_in = domain_train.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
            #value = domain_in[:, :self.args.seq_len, :]
            #label = domain_in[:, self.args.seq_len:, :-2]
            #domain_test_dataset = DomainDataset(value, label)
            #domain_test_loader = torch.utils.data.DataLoader(domain_test_dataset, batch_size=1, shuffle=False,
                                                                 #drop_last=True)
            for batch in dataloader:
                x = batch[0].to('cuda')
                y = batch[1]
                outputs = self.model(x)
                s_loss = criterion(outputs.to(y.device), y)
                if (k + 1) % 10000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(k + 1, epoch + 1, s_loss.item()))
                k = k + 1
                model_optim.zero_grad()
                s_loss.backward()
                model_optim.step()
            '''
            for batch in domain_test_loader:
                x = batch[0].to('cuda')
                y = batch[1]
                outputs = self.model(x)
                t_loss = criterion(outputs.to(y.device), y)
                if (k + 1) % 1000 == 0:
                    print("\ttarget: iters: {0}, epoch: {1} | loss: {2:.7f}".format(k + 1, epoch + 1, t_loss.item()))
                k = k + 1
                model_optim.zero_grad()
                t_loss.backward()
                model_optim.step()
            '''

        return self.model

    def vali(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))
        vali_domain_len = int(0.2 * len(data))  # 100
        r_avg = []
        m_avg = []
        rmse = nn.MSELoss(reduction="mean")
        mae = nn.L1Loss()
        for domain in range(vali_domain_len):
            domain_vali = torch.tensor(data[domain_len + 1 + domain]).float()
            vali_in = domain_vali.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
            vali_value = vali_in[:, :self.args.seq_len, :]
            vali_label = vali_in[:, self.args.seq_len:, :-2]
            vali_dataset = DomainDataset(vali_value, vali_label)
            valiloader = torch.utils.data.DataLoader(vali_dataset, batch_size=1,
                                                     shuffle=False, drop_last=True)
            self.model.eval()
            avg_r_loss = []
            avg_m_loss = []
            for batch in valiloader:
                x = batch[0].to('cuda')
                y = batch[1]
                outputs = self.model(x)
                m_loss = mae(outputs.to(y.device), y)
                r_loss = rmse(outputs.to(y.device), y)
                avg_m_loss.append(m_loss.detach().numpy())
                avg_r_loss.append(r_loss.detach().numpy())
            avg_m_loss = np.array(avg_m_loss).mean()
            avg_r_loss = np.array(avg_r_loss).mean()
            m_avg.append(avg_m_loss)
            r_avg.append(avg_r_loss)
        m_avg = np.array(m_avg).mean()
        r_avg = np.array(r_avg).mean()
        print("测试集上的MAE:{0}".format(m_avg))
        print("测试集上的RMSE:{0}".format(r_avg))
        return

    def test(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))
        vali_domain_len = int(0.2 * len(data))
        test_domain_len = int(0.2*len(data))#100
        r_avg = []
        m_avg = []
        rmse = nn.MSELoss(reduction="mean")
        mae = nn.L1Loss()
        for domain in range(test_domain_len):
            domain_test = torch.tensor(data[domain_len + vali_domain_len + 1 + domain]).float()
            test_in = domain_test.unfold(0, self.args.seq_len+self.args.pred_len, 1).transpose(-1, 1)
            test_value = test_in[:, :self.args.seq_len, :]
            test_label = test_in[:, self.args.seq_len:, :-2]
            test_dataset = DomainDataset(test_value, test_label)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False, drop_last=True)
            self.model.eval()
            avg_r_loss = []
            avg_m_loss = []
            for batch in testloader:
                x = batch[0].to('cuda')
                y = batch[1]
                outputs = self.model(x)
                m_loss = mae(outputs.to(y.device), y)
                r_loss = rmse(outputs.to(y.device), y)
                avg_m_loss.append(m_loss.detach().numpy())
                avg_r_loss.append(r_loss.detach().numpy())
            avg_m_loss = np.array(avg_m_loss).mean()
            avg_r_loss = np.array(avg_r_loss).mean()
            m_avg.append(avg_m_loss)
            r_avg.append(avg_r_loss)
        m_avg = np.array(m_avg).mean()
        r_avg = np.array(r_avg).mean()
        print("测试集上的MAE:{0}".format(m_avg))
        print("测试集上的RMSE:{0}".format(r_avg))
        return