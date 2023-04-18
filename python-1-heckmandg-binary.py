#%% Modules
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

data_name = 'camelyon17'  #'insight', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
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
from networks import HeckmanDNN, HeckmanCNN 
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


#%% HeckmanDG_CNN_Regressor
import tqdm
import torch.nn.functional as F
from utils_datasets.transforms import InputTransforms
optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
scheduler = None

from networks import HeckmanDNN, HeckmanCNN 
from models.heckmandg_regression import HeckmanDG_CNN_Regressor
from models.heckmandg_binaryclass import HeckmanDG_CNN_BinaryClassifier
from models.heckmandg_multiclass import HeckmanDG_CNN_MultiClassifier
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
model = HeckmanDG_CNN_BinaryClassifier(args, network, optimizer)
model.fit(train_loader, id_valid_loader, ood_valid_loader)
torch.save(model.network.state_dict(), f'./results/{args.data}_{args.seed}_model.pth')


if False:
    # load the model
    model = HeckmanDGBinaryClassifierCNN()
    # Load the saved model state dictionary
    state_dict = torch.load(f'./results/{args.data}_{args.seed}_model.pth')
    # Load the saved state dictionary into the model
    model.load_state_dict(state_dict)
    
#%% 4. Results Analysis
from utils.result import prediction, plots_loss, plots_probit

# prediction results
res_tr = prediction(train_loader, model, args)
res_vl_id = prediction(id_valid_loader, model, args)
res_vl_ood = prediction(ood_valid_loader, model, args)
res_ts = prediction(test_loader, model, args)

res_tr_mean = pd.DataFrame(res_tr).mean().round(3)
res_id_vl_mean = pd.DataFrame(res_vl_id).mean().round(3)
res_ood_vl_mean = pd.DataFrame(res_vl_ood).mean().round(3)
res_ts_mean = pd.DataFrame(res_ts).mean().round(3)

results = np.concatenate((np.array([f'{args.data}']), res_tr_mean.values, res_id_vl_mean.values, res_ood_vl_mean.values, res_ts_mean.values))
results = pd.DataFrame([results])
results.columns = ['data', 
                   'train_auc', 'train_f1', 'train_acc',
                   'valid_id_auc', 'valid_id_f1', 'valid_id_acc',
                   'valid_ood_auc', 'valid_ood_f1', 'valid_ood_acc',
                   'test_auc', 'test_f1', 'test_acc']
print(results)
results.to_csv(f'./results/prediction/HeckmanDG_{args.data}.csv')

# plots: loss, probits
domain_color_map = {
    0: 'orange',
    1: 'slateblue',
    2: 'navy',
    3: 'crimson',
    4: 'darkgreen',
}
plots_loss(model, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_loss.pdf")
probits, labels = model.get_selection_probit(train_loader)
plots_probit(probits, labels, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_probits.pdf")

    
