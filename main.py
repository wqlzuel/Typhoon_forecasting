import argparse
import os
import torch
from exp_main import Exp_Main
import random
import numpy as np

def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time Series Forecasting Methods for Typhoon track forecasting')
    parser.add_argument('--model', type=str, required=True, default='Attention',
                        help='model name, options: [Dlinear, Attention, Transformer]')
    parser.add_argument('--seq_len', type=int, default=6, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='optimizer learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='device ids of multile gpus')

    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main
    exp = Exp(args)
    exp.train()
    exp.vali()
    exp.test()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()