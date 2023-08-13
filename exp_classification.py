from models import Dlinear, Attention, Transformer, DomainNet
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from exp_basic import Exp_Basic, DomainDataset

def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

class Exp_Main_C(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_C, self).__init__(args)
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
                d = [int(s[3])] + [int(s[4])] + [int(s[5])] + [int(s[2])]
                series.append(d)
            else:
                if len(series) > (min_len - 1):  # len(series)>0
                    data.append(series)
                    series = []
        return data

    def _get_data(self):
        read_data = pd.read_csv('/typhoon/data/bst_all.txt', sep='\t', header=None)
        data = self.preprocess(read_data)
        return data

    def _select_criterion(self):
        criterion = nn.MSELoss()#nn.CrossEntropyLoss()
        return criterion

    def train(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))  # 300##1000  # len(data)#域个数
        criterion = self._select_criterion()
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        k = 0
        self.model.train()
        domain_train = torch.tensor(data[0])
        domain_in = domain_train.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
        value = domain_in[:, :self.args.seq_len, :]
        label = domain_in[:, self.args.seq_len:, -1]
        for domain in range(domain_len):
            domain_train = torch.tensor(data[domain]).float()
            domain_in = domain_train.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)  # 10为seq_len+pred_len
            value = torch.concat([value, domain_in[:, :self.args.seq_len, :]], dim=0)
            label = torch.concat([label, domain_in[:, self.args.seq_len:, -1]], dim=0)
        domain_dataset = DomainDataset(value, label)
        dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=32, shuffle=False, drop_last=False)
        train_epoch = self.args.train_epochs  # 200
        for epoch in range(train_epoch):  # 20 10 30
            for batch in dataloader:
                x = batch[0].to('cuda')
                y = batch[1]#.long()-2#1#.unsqueeze(-1).long()
                outputs = self.model(x)
                loss = criterion(outputs.to(y.device), y)
                if (k + 1) % 10000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(k + 1, epoch + 1, loss.item()))
                k = k + 1
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

        return self.model
    def vali(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))
        vali_domain_len = int(0.2 * len(data))  # 100
        avg = []
        mae = nn.L1Loss(reduction = 'sum')#nn.CrossEntropyLoss()
        for domain in range(vali_domain_len):
            domain_vali = torch.tensor(data[domain_len + 1 + domain]).float()
            vali_in = domain_vali.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
            vali_value = vali_in[:, :self.args.seq_len, :]
            vali_label = vali_in[:, self.args.seq_len:, -1]
            vali_dataset = DomainDataset(vali_value, vali_label)
            valiloader = torch.utils.data.DataLoader(vali_dataset, batch_size=1,
                                                     shuffle=False, drop_last=True)
            self.model.eval()
            avg_loss = []
            for batch in valiloader:
                x = batch[0].to('cuda')
                y = batch[1]#.long()-2#1
                outputs = self.model(x).int().float()
                loss = mae(outputs.to(y.device), y)#cal_accuracy(torch.argmax(outputs.transpose(-1, 1), dim=-1).detach().cpu().numpy(), y.detach().numpy())#acc(outputs.to(y.device), y)
                #print(loss)
                avg_loss.append(loss.detach().numpy())#.append(loss)
            avg_loss = np.array(avg_loss).mean()
            avg.append(avg_loss)
        avg = np.array(avg).mean()
        print("测试集上的准确率为:{0}".format(avg))
        return

    def test(self):
        data = self._get_data()
        domain_len = int(0.6 * len(data))
        vali_domain_len = int(0.2 * len(data))
        test_domain_len = int(0.2 * len(data))  # 100
        avg = []
        mae = nn.L1Loss(reduction = 'sum')
        for domain in range(test_domain_len):
            domain_test = torch.tensor(data[domain_len + vali_domain_len + 1 + domain]).float()
            test_in = domain_test.unfold(0, self.args.seq_len + self.args.pred_len, 1).transpose(-1, 1)
            test_value = test_in[:, :self.args.seq_len, :]
            test_label = test_in[:, self.args.seq_len:, -1]
            test_dataset = DomainDataset(test_value, test_label)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                     shuffle=False, drop_last=True)
            self.model.eval()
            avg_loss = []
            for batch in testloader:
                x = batch[0].to('cuda')
                y = batch[1]
                outputs = self.model(x).int().float()
                loss = mae(outputs.to(y.device), y)
                avg_loss.append(loss.detach().numpy())
            avg_loss = np.array(avg_loss).mean()
            avg.append(avg_loss)
        avg = np.array(avg).mean()
        print("验证集上的MAE:{0}".format(avg))
        return