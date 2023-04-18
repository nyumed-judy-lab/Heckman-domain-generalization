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
from utils.argparser import DatasetImporter, parse_arguments
from utils.argparser import args_insight, args_cameloyn17, args_poverty, args_iwildcam, args_rxrx1
from utils.dataloader import dataloaders, sub_dataloaders
from utils.seed import fix_random_seed

os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)
os.makedirs('./results/prediction/', exist_ok=True)

#%% 1. Experiment Settings
def data_argument(args, data_name, fold: str=None):
    if data_name == 'insight':
        args = args_insight()
    elif data_name == 'camelyon17':
        args = args_cameloyn17(args) # data-specific arguments 
    elif data_name == 'poverty':
        args = args_poverty(args) # data-specific arguments 
        args.fold_list = ['A', 'B', 'C', 'D', 'E']        
        if fold is not None:
            args.fold = fold
        else:
            args.fold_list[0] #fold #'A'
    elif data_name == 'rxrx1':
        args = args_rxrx1(args) # data-specific arguments 
    elif data_name == 'iwildcam':
        args = args_iwildcam(args) # data-specific arguments 
    else:
        print("choose the data_name among ('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam')")
    return args

data_list = ['insight', 'camelyon17', 'iwildcam', 'poverty', 'rxrx1']

if True:
    data_name = 'poverty'
    experiment_name = 'Heckman DG'
    # basic arguments for image data
    args = parse_arguments(experiment_name) 
    # Data-Specific arguments
    fold = None
    if data_name=='poverty':
        poverty_folds = ['A', 'B', 'C', 'D', 'E']
        # select specific fold in the poverty
        fold = poverty_folds[0]
    args = data_argument(args, data_name, fold)
    dataset = DatasetImporter(args)

fix_random_seed(args.seed)

#%% 2. Data Preparation
from utils.argparser import DatasetImporter
from utils.preprocessing import preprocessing_tabular

if args.data_type == 'tabular':
    # dat = pd.read_feather(args.root)
    covariate_list = dataset.columns.drop(['SSID', 'SITE', 'CVD'])
    # set the name_list of numeric (continuous variables 
    num_prefix = ['AGE', 'LAB', 'ENC', 'FLWUP', 'VITAL_HT', 'VITAL_WT', 'VITAL_DIASTOLIC', 'VITAL_SYSTOLIC', 'VITAL_BMI'] #'VITAL', 
    num_cols = [c for c in covariate_list if any(c.startswith(prefix) for prefix in num_prefix)]
    cat_cols = [c for c in covariate_list if not any(c.startswith(prefix) for prefix in num_prefix)]
    # DOMAIN-GENERALIZATION EXPERIEMNT SETTING
    domains = dataset['SITE'].unique().tolist()
    
    domain_color_map = {
    'A05': 'orange',
    'A07': 'navy',
    'B03': 'darkgreen',
    'C02': 'slateblue',
    'C05': 'crimson'
    }
    # set the train domains
    train_domains = ['A05', 'A07', 'B03', 'C05']
    args.train_domains = train_domains
    args.num_domains = len(train_domains)
    train_val_dat, test_dat = train_test_split(dataset, stratify=dataset['SITE']+dataset['CVD'].astype(str), test_size=args.test_size, random_state=args.seed)
    tr_x, tr_s, tr_y, val_x, val_s, val_y, num_imputer, scaler = preprocessing_tabular(train_val_dat, args, num_cols, cat_cols)

elif args.data_type == 'image':
    # (1) run the experiment with all data to test the implementation of HeckmanDG (take large amount of memory)
    train_loader, valid_loader, test_loader = dataloaders(args, dataset)
    # (2) set TRUE if you want to run the experiment with subset of data to test the implementation of HeckmanDG (take small amount of memory)
    if True:
        train_loader, valid_loader, test_loader = sub_dataloaders(train_loader, valid_loader, test_loader)

#%% 3. HeckmanDG
from networks import HeckmanDNN, HeckmanCNN 
from models import HeckmanDG_DNN_BinaryClassifier
from models import HeckmanDG_CNN_BinaryClassifier
from models import HeckmanDG_CNN_Regressor
from models import HeckmanDG_CNN_MultiClassifier

from utils_datasets.transforms import InputTransforms

if args.data_type == 'tabular':
    network = HeckmanDNN([tr_x.shape[1], 128, 64, 32, 1], [tr_x.shape[1], 64, 32, 16, args.num_domains], dropout=0.5, batchnorm=True, activation='ReLU')
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 75, 100, 125], gamma=.1)
    model = HeckmanDG_DNN_BinaryClassifier(network, optimizer, scheduler, config={'max_epoch': args.epochs})
    # model = HeckmanDGBinaryClassifier(network, optimizer, scheduler, config={'max_epoch': 10})
    model.fit(
        {'train_x': tr_x,
         'train_y': tr_y,
         'train_s': pd.get_dummies(tr_s).values,
         'valid_x': val_x,
         'valid_y': val_y,
         'valid_s': pd.get_dummies(val_s).values})
    
elif args.data_type == 'image':
    network = HeckmanCNN(args)

    scheduler = None
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

    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.loss_type == 'binary':
        model = HeckmanDG_CNN_BinaryClassifier(args, network, optimizer, scheduler)
    elif args.loss_type == 'regression':
        model = HeckmanDG_CNN_Regressor(args, network, optimizer, scheduler)
    elif args.loss_type =='multiclass':
        model = HeckmanDG_CNN_MultiClassifier(args, network, optimizer, scheduler)
        
        
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
if args.data_type=='tabular':
    all, internal, external = [], [], []
    for site in domains:
        print(site)
        te_x = test_dat[test_dat['SITE'] == site][num_cols].values.astype(float)
        te_y = test_dat[test_dat['SITE'] == site]['CVD'].values.astype(float)
        te_x = num_imputer.transform(te_x)
        te_x = scaler.transform(te_x)
        te_x = np.column_stack([te_x, test_dat[test_dat['SITE'] == site][cat_cols].values.astype(float)])
        score = roc_auc_score(te_y, model.predict_proba(te_x)).round(3)
        if site in train_domains:
            internal.append(score)
        else:
            external.append(score)
        all.append(score)
    results = pd.DataFrame([all + [np.mean(internal), np.mean(external), np.mean(all)]],
                       index=['+'.join(train_domains)], 
                       columns=domains + ['Internal', 'External', 'All'])
    results.to_csv(f'./results/prediction/HeckmanDG_{args.data}.csv')
    

elif args.data_type=='image':
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