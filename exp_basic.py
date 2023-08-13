import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model = self._build_model().to(args.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

class DomainDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X  # set data
        self.Y = Y  # set lables

    def __len__(self):
        return len(self.X)  # return length

    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx]]