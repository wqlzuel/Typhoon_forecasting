import argparse
import os
import torch
from exp_forecast import Exp_Main_F
from exp_classification import Exp_Main_C
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
    parser.add_argument('--seq_len', type=int, default=6, help='input sequence length')#6
    parser.add_argument('--pred_len', type=int, default=4, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')  # wth:21 #traffic:862 #electricity:321
    parser.add_argument('--dec_in', type=int, default=4, help='decoder input size')#3
    parser.add_argument('--c_out', type=int, default=3, help='output size')#3 4
    parser.add_argument('--d_model', type=int, default=6, help='dimension of model')  #6 512
    parser.add_argument('--n_heads', type=int, default=2, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')  # 1
    parser.add_argument('--d_ff', type=int, default=6, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--train_epochs', type=int, default=600, help='train epochs')#600 200
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.00025, help='optimizer learning rate')#0.00025
    parser.add_argument('--device', type=str, default='cuda', help='device ids of multile gpus')
    parser.add_argument('--task', type=str, default='f', help='')

    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    if args.task == 'f':
        Exp = Exp_Main_F
    else:
        Exp = Exp_Main_C
    #Exp = Exp_Main
    exp = Exp(args)
    exp.train()
    exp.vali()
    exp.test()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()