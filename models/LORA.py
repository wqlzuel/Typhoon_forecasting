import torch
import torch.nn as nn
from math import sqrt
class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoraLinear, self).__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.lora_A = nn.Parameter(torch.FloatTensor(in_features, r))
        stdv = 1. / sqrt(self.lora_A.size(1))
        self.lora_A.data.uniform_(-stdv, stdv)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.Linear = nn.Linear(in_features, out_features)
    def forward(self, x):
        result = self.Linear(x)
        lora_result = torch.matmul(x, self.lora_A)
        lora_result = torch.matmul(lora_result, self.lora_B)
        final_result = result + lora_result
        return final_result