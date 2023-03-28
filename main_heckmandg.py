#%% Modules
import os
os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)
os.makedirs('./results/prediction/', exist_ok=True)

import torch
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from functools import partial

def fix_random_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)

#%% 1. Experiment Settings
from utils_datasets.defaults import DataDefaults
from utils.argparser import DatasetImporter, parse_arguments, args_cameloyn17_outcome
from utils.dataloader import dataloaders, sub_dataloaders

data_name = 'camelyon17'
data_type = 'image'
experiment_name = 'Heckman DG Benchmark'

args = parse_arguments(experiment_name) # basic arguments for image data
args = args_cameloyn17_outcome(args, experiment_name) # data-specific arguments 
# DataDefaults: has data-specific hyperparameters
defaults = DataDefaults[args.data]() 
args.num_domains = len(defaults.train_domains)
args.num_classes = defaults.num_classes
args.loss_type = defaults.loss_type
args.train_domains = defaults.train_domains
args.validation_domains = defaults.validation_domains
args.test_domains = defaults.test_domains

fix_random_seed(args.seed)    

#%% 2. Data Preparation
# DatasetImporter: put DataDefaults into DatasetImporter to get the dataset
dataset = DatasetImporter(defaults, args)

# (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
train_loader, valid_loader, test_loader = dataloaders(args, dataset)

'''
# (2) run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
if True:
    train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)
'''

#%% 3. HeckmanDG
from networks import SeparatedHeckmanNetwork, SeparatedHeckmanNetworkCNN # 
from models import HeckmanDGBinaryClassifier, HeckmanDGBinaryClassifierCNN # 

if data_type == 'tabular':
    # len(train_loader.dataset['x'].shape)>4
    network = SeparatedHeckmanNetwork(args)
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
    model = HeckmanDGBinaryClassifier(args, network, optimizer, scheduler)
    model.fit(train_loader, valid_loader)
elif data_type == 'image':
    # len(train_loader.dataset['x'].shape)<4
    network = SeparatedHeckmanNetworkCNN(args)
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
    model = HeckmanDGBinaryClassifierCNN(args, network, optimizer, scheduler)
    model.fit(train_loader, valid_loader)
    
#%% 4. Results Analysis
# plots: loss, probits
from utils.plots import plots_loss, plots_probit
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

# prediction results
res_tr = []
res_vl = []
res_ts = []
for b, batch in enumerate(train_loader):
    print(f'train_loader {b} batch / {len(train_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_tr.append(score)
    except ValueError:
        pass
for b, batch in enumerate(valid_loader):
    print(f'valid_loader {b} batch / {len(valid_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_vl.append(score)
    except ValueError:
        pass
for b, batch in enumerate(test_loader):
    print(f'{b} batch / {len(test_loader)}')
    y_true = batch['y']
    y_pred = model.predict_proba(batch)
    try:
        score = roc_auc_score(y_true, y_pred)
        res_ts.append(score)
    except ValueError:
        pass
    
res_tr_mean = pd.DataFrame(res_tr).mean()
res_vl_mean = pd.DataFrame(res_vl).mean()
res_ts_mean = pd.DataFrame(res_ts).mean()

results = pd.concat([pd.DataFrame([args.data]), res_tr_mean, res_vl_mean, res_ts_mean], axis=1)
results.columns = ['data', 'train', 'valid', 'test']
print(results)
results.to_csv(f'./results/prediction/HeckmanDG_{args.data}.csv')

"""
python 2.run-cameloyon17-CNN-OneStep-HeckmanDG.py
"""
