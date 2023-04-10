#%% Modules
import os
import os
import torch
import argparse
import numpy as np
import pandas as pd

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

if False:
    data_name_list = ['insight', 'camelyon17', 'iwildcam', 'poverty', 'rxrx1']
    data_name_list = ['rxrx1']
    for data_name in data_name_list:
        # data_name = 'rxrx1',  #'insight', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
        experiment_name = 'Heckman DG'
        args = parse_arguments(experiment_name) # basic arguments for image data
        args = data_argument(args, data_name)
        dataset = DatasetImporter(args)
        args.num_domains = len(dataset.train_domains) 
        print(args.num_domains)

data_name = 'poverty'  #'insight', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
experiment_name = 'Heckman DG'
args = parse_arguments(experiment_name) # basic arguments for image data
args = data_argument(args, data_name)
dataset = DatasetImporter(args)

# (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
train_loader, valid_loader, test_loader = dataloaders(args, dataset)
# (2) run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
if False:
    train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)

#%% Intialize network, models
from networks import HeckmanDNN, HeckmanCNN 
import torch.nn.functional as F
from models.heckmandg_regression import HeckmanDG_CNN_Regressor
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
model = HeckmanDG_CNN_Regressor(args, network, optimizer)
model.fit(train_loader, valid_loader)

torch.save(model.network.state_dict(), f'./results/{args.data}_{args.seed}_model.pth')

#%% 4. Results Analysis
from utils.result import prediction, plots_loss, plots_probit

res_tr = prediction(train_loader, model, args)
res_vl = prediction(valid_loader, model, args)
res_ts = prediction(test_loader, model, args)

res_tr_mean = pd.DataFrame(res_tr).mean().round(3)
res_vl_mean = pd.DataFrame(res_vl).mean().round(3)
res_ts_mean = pd.DataFrame(res_ts).mean().round(3)

results = np.concatenate((np.array([f'{args.data}']), res_tr_mean.values, res_vl_mean.values, res_ts_mean.values))
results = pd.DataFrame([results])
if args.loss_type == 'regression':
    results.columns = ['data', 
                    'train_mse', 'train_mae', 'train_pearson',
                    'valid_mse', 'valid_mae', 'valid_pearson',
                    'test_mse', 'test_mae', 'test_pearson']
else:
    results.columns = ['data', 
                        'train_auc', 'train_f1', 'train_acc',
                        'valid_auc', 'valid_f1', 'valid_acc',
                        'test_auc', 'test_f1', 'test_acc']

print(results)
if args.data == 'poverty':
    results.to_csv(f'./results/prediction/HeckmanDG_{args.data}_{args.fold}.csv')
else:
    results.to_csv(f'./results/prediction/HeckmanDG_{args.data}.csv')
