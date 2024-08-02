import os
import random
import numpy as np
import torch
import argparse
import pickle
import pandas as pd
from train_test_cell import train_cell


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Configurations for Cell_cls Training')
parser.add_argument('--data_root_dir', type=str, default=None)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--lr_policy', default='linear', type=str)
parser.add_argument('--lr', default=2e-3, type=float)
parser.add_argument('--weight_decay', default=4e-4, type=float)
args = parser.parse_args()

cindex_folds=[]
for i in args.folds:
    train_cell(args, i,device)

