#%% Modules
import os
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from functools import partial
from itertools import combinations
import matplotlib.pyplot as plt

#%% 1. Experiment Settings
# from utils_datasets.defaults import DataDefaults
from utils.argparser import DatasetImporter, parse_arguments
from utils.argparser import args_insight
from utils.argparser import args_cameloyn17, args_poverty, args_iwildcam, args_rxrx1
from utils.dataloader import dataloaders, sub_dataloaders
from utils.seed import fix_random_seed

os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)
os.makedirs('./results/prediction/', exist_ok=True)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#%% 1. Experiment Settings
# from utils_datasets.defaults import DataDefaults

def data_argument(args, data_name):
    if data_name == 'insight':
        args = args_insight()
    if data_name == 'camelyon17':
        args = args_cameloyn17(args) # data-specific arguments 
    elif data_name == 'poverty':
        args = args_poverty(args) # data-specific arguments 

    elif data_name == 'rxrx1':
        args = args_rxrx1(args) # data-specific arguments 
    elif data_name == 'iwildcam':
        args = args_iwildcam(args) # data-specific arguments 
    else:
        print("choose the data_name among ('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam')")
    return args


data_name = 'iwildcam'  #'insight', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
experiment_name = 'Heckman DG'
args = parse_arguments(experiment_name) # basic arguments for image data
args = data_argument(args, data_name)

import pickle
if False:
    dataset = DatasetImporter(args)
    # save the batch using pickle
    with open(f'./data/examples/dataset_{args.data}.pkl', 'wb') as f:
        pickle.dump(dataset, f)
if True:        
    with open(f'./data/examples/dataset_{args.data}.pkl', 'rb') as f:
        dataset = pickle.load(f)    

    # (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
    train_loader, id_valid_loader, ood_valid_loader, test_loader = dataloaders(args, dataset)
    # (2) run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
    if True:
        train_loader, id_valid_loader, ood_valid_loader, test_loader = sub_dataloaders(train_loader, id_valid_loader, ood_valid_loader, test_loader)
if False:
    batch = next(iter(train_loader))
    # save the batch using pickle
    with open(f'./data/examples/batch_{args.data}.pkl', 'wb') as f:
        pickle.dump(batch, f)
if True:
    with open(f'./data/examples/batch_{args.data}.pkl', 'rb') as f:
        batch = pickle.load(f)

setattr(args, 'train_domains', dataset.train_domains)
setattr(args, 'validation_domains', dataset.validation_domains)
setattr(args, 'test_domains', dataset.test_domains)
setattr(args, 'num_domains', len(dataset.train_domains))

#%% Intialize network, models
from networks import HeckmanCNN
import torch.nn.functional as F
network = HeckmanCNN(args)

if args.loss_type == 'regression':
    param_groups = [
        {'params': network.f_layers.parameters()},
        {'params': network.g_layers.parameters()},
        {'params': network.rho},
        {'params': network.sigma}
    ]
else:
    param_groups = [
    {'params': network.f_layers.parameters()},
    {'params': network.g_layers.parameters()},
    {'params': network.rho}
    ]
optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)

from models.heckmandg_multiclass import HeckmanDG_CNN_MultiClassifier
model = HeckmanDG_CNN_MultiClassifier(args, network, optimizer)
model.fit(train_loader, valid_loader)
