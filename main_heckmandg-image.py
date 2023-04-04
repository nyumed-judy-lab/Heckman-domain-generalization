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
from functools import partial
from itertools import combinations
import matplotlib.pyplot as plt

#%% 1. Experiment Settings
# from utils_datasets.defaults import DataDefaults
from utils.argparser import DatasetImporter, parse_arguments
from utils.argparser import args_cameloyn17, args_poverty, args_iwildcam, args_rxrx1
from utils.dataloader import dataloaders, sub_dataloaders
from utils.argparser import args_insight
from utils.seed import fix_random_seed

os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)
os.makedirs('./results/prediction/', exist_ok=True)

data_name = 'camelyon17' #'insight', 'camelyon17', 'iwildcam', 'iwildcam', 'iwildcam', 'iwildcam'
experiment_name = 'Heckman DG'
args = parse_arguments(experiment_name) # basic arguments for image data

if data_name == 'camelyon17':
    data_type = 'image'
    args = args_cameloyn17(args) # data-specific arguments 
elif data_name == 'poverty':
    data_type = 'image'
    args = args_poverty(args) # data-specific arguments 
elif data_name == 'rxrx1':
    data_type = 'image'
    args = args_rxrx1(args) # data-specific arguments 
elif data_name == 'iwildcam':
    data_type = 'image'
    args = args_iwildcam(args) # data-specific arguments 
else:
    print("choose the data_name among ('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam')")

fix_random_seed(args.seed)

#%% 2. Data Preparation
dataset = DatasetImporter(args)
# (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
train_loader, valid_loader, test_loader = dataloaders(args, dataset)
# (2) run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
if False:
    train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)

#%% 3. HeckmanDG
from networks import HeckmanDNN, HeckmanCNN 
from models import HeckmanDGBinaryClassifier, HeckmanDGBinaryClassifierCNN
from utils_datasets.transforms import InputTransforms

network = HeckmanCNN(args)
optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
model = HeckmanDGBinaryClassifierCNN(args, network, optimizer, scheduler)
model.fit(train_loader, valid_loader)
# save the model
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
res_vl = prediction(valid_loader, model, args)
res_ts = prediction(test_loader, model, args)

res_tr_mean = pd.DataFrame(res_tr).mean().round(3)
res_vl_mean = pd.DataFrame(res_vl).mean().round(3)
res_ts_mean = pd.DataFrame(res_ts).mean().round(3)

results = np.concatenate((np.array([f'{args.data}']), res_tr_mean.values, res_vl_mean.values, res_ts_mean.values))
results = pd.DataFrame([results])
results.columns = ['data', 
                   'train_auc', 'train_f1', 'train_acc',
                   'valid_auc', 'valid_f1', 'valid_acc',
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

    
"""
python main_heckmandg.py
"""
