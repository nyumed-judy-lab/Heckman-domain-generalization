#%% Modules
import sys
import numpy
import torch
import numpy as np
import pandas as pd
import argparse
import random
import warnings
import argparse
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from functools import partial
from itertools import combinations

def fix_random_seed(s: int):
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    
#%% 1. Data Input
from utils_datasets.defaults import DataDefaults
from utils.argparser import parse_arguments, parse_arguments_outcome
from utils.argparser import args_cameloyn17, args_cameloyn17_outcome, args_poverty, args_poverty_outcome
from utils.argparser import DatasetImporter
from utils.dataloader import dataloaders

dataname = 'camelyon17'
experiment_name = 'Heckman DG Benchmark'
# experiemnt configuration
# parser = argparse.ArgumentParser()
args = parse_arguments(experiment_name)
args = args_cameloyn17_outcome(args, experiment_name) # args = parse_arguments(experiment_name)

fix_random_seed(args.seed)

# DataDefaults: has data-specific hyperparameters
defaults = DataDefaults[args.data]() 
args.num_domains = len(defaults.train_domains)
args.num_classes = defaults.num_classes
args.loss_type = defaults.loss_type
args.train_domains = defaults.train_domains
args.validation_domains = defaults.validation_domains
args.test_domains = defaults.test_domains
# DatasetImporter: put DataDefaults into DatasetImporter to get the dataset
dataset = DatasetImporter(defaults, args)
# Data Loader
train_loader, valid_loader, test_loader = dataloaders(args, dataset)


#%% Intialize network, models
from networks import SeparatedHeckmanNetworkCNN #SeparatedHeckmanNetwork, 
from models import HeckmanDGBinaryClassifierCNN #HeckmanDGBinaryClassifier, 
network = SeparatedHeckmanNetworkCNN(args)
optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[2, 4], gamma=.1)
model = HeckmanDGBinaryClassifierCNN(args, network, optimizer, scheduler)
model.fit(train_loader, valid_loader)

opt = optimizer(
    [{'params': network.f_layers.parameters()},
     {'params': network.g_layers.parameters()},
     {'params': network.rho, 'lr': 1e-2, 'weight_decay': 0.}]
)
sch = scheduler(opt) 

train_loss_traj, valid_loss_traj = [], []
train_auc_traj, valid_auc_traj = [], []
rho_traj = []
best_model, best_loss, best_acc = deepcopy(network.state_dict()), 1e10, 0.
normal = torch.distributions.normal.Normal(0, 1)
config = {'device': 'cuda',
          'max_epoch': args.epochs, #self.args.epochs
          'batch_size': args.batch_size #self.args.batch_size
        }
config.update(config)

pbar = tqdm(range(args.epochs))


# heckmandg for CNN with images
for epoch in pbar:
    # print(epoch)
    # self.network.train()
    network.train() #self
    train_loss = 0.
    train_pred, train_true = [], []
    ##### train_dataloader
    # for batch in dataloaders['train']:
    for batch in train_loader:
        # print(batch)
        # x_batch = [b.to(config['device']) for b in batch['x']]
        # x_batch.to(torch.float32)
        
        x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
        y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
        s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
        # s_ = [str(_.item()) for _ in s]
        import numpy as np
        one_hot = np.zeros((args.batch_size, len(defaults.train_domains)))
        col_idx = [np.where(int(s_[idx].item())  == np.array(defaults.train_domains))[0][0] for idx in range(len(s_))]
        for i in range(len(one_hot)):
            one_hot[i,col_idx[i]] = 1
        s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)

        batch_probits = network(x) # shape(batch, output of f_layers+g_layers) , 
        batch_prob = normal.cdf(batch_probits[:, 0]) # 0: f_layerrs
        # network.forward(x)
        # network.forward_g(x)
        # network.forward_f(x) 
        
        safe_log = lambda x: torch.log(torch.clip(x, 1e-6))
        loss = 0.
        for j in range(args.num_domains):
            from models.heckmandg import bivariate_normal_cdf
            joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j + 1], network.rho[j]) #self.
            loss += -(y * s[:, j] * safe_log(joint_prob)).mean()
            loss += -((1 - y) * s[:, j] * safe_log(
                normal.cdf(batch_probits[:, j + 1]) - joint_prob)).mean()
            loss += -((1 - s[:, j]) * safe_log(normal.cdf(-batch_probits[:, j + 1]))).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        network.rho.data = torch.clip(network.rho.data, -0.99, 0.99)

        train_loss += loss.item() / len(train_loader)
        train_pred.append(batch_prob.detach().cpu().numpy())
        train_true.append(y.detach().cpu().numpy())

    train_loss_traj.append(train_loss)
    train_auc_traj.append(
        roc_auc_score(
            np.concatenate(train_true),
            np.concatenate(train_pred)))

    rho_traj.append(network.rho.data.detach().cpu().numpy())

    if sch:
        sch.step()

    network.eval()
    # ID validation
    with torch.no_grad():
        valid_loss = 0.
        valid_pred, valid_true = [], []
        ##### valid_dataloader
        for batch in valid_loader:
            # print(batch)
            x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
            y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
            s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)

            one_hot = np.zeros((args.batch_size, len(defaults.train_domains)))
            col_idx = [np.where(int(s_[idx].item())  == np.array(defaults.train_domains))[0][0] for idx in range(len(s_))]
            for i in range(len(one_hot)):
                one_hot[i,col_idx[i]] = 1
            s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)

            # batch = [b.to(self.config['device']) for b in batch]
            batch_probits = network(x)
            batch_prob = normal.cdf(batch_probits[:, 0])

            loss = 0.
            for j in range(args.num_domains):
                joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], network.rho[j])
                loss += -(y * s[:, j] * safe_log(joint_prob)).mean()
                loss += -((1 - y) * s[:, j] * safe_log(normal.cdf(batch_probits[:, j+1]) - joint_prob)).mean()
                loss += -((1 - s[:, j]) * safe_log(1. - normal.cdf(batch_probits[:, j+1]))).mean()

            valid_loss += loss.item() / len(valid_loader)
            valid_pred.append(batch_prob.detach().cpu().numpy())
            valid_true.append(y.detach().cpu().numpy())

    if valid_loss < best_loss:
        best_model = deepcopy(network.state_dict())
        best_loss = valid_loss
        best_epoch = epoch

    valid_loss_traj.append(valid_loss)
    valid_auc_traj.append(
        roc_auc_score(
            np.concatenate(valid_true),
            np.concatenate(valid_pred)))

    desc = f'[{epoch + 1:03d}/{config["max_epoch"]:03d}] '
    desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
    desc += f'AUROC: {train_auc_traj[-1]:.4f} '
    desc += f'| valid | Loss: {valid_loss_traj[-1]:.4f} '
    desc += f'AUROC: {valid_auc_traj[-1]:.4f} '
    pbar.set_description(desc)

##### predict_prob
normal = torch.distributions.normal.Normal(0, 1)

probs_list=[]
for batch in train_loader:
    x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
    y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
    s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)

    with torch.no_grad():
        probs_batch = normal.cdf(network(x.to(args.device))[:, 0])
        probs_list.append(probs_batch)
        
probs = torch.cat(probs_list).detach().cpu().numpy()
##################################################
##### get_selection_prob
probits_list = []
for batch in train_loader:
    x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
    y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
    s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
    one_hot = np.zeros((args.batch_size, len(defaults.train_domains)))
    col_idx = [np.where(int(s_[idx].item())  == np.array(defaults.train_domains))[0][0] for idx in range(len(s_))]
    for i in range(len(one_hot)):
        one_hot[i,col_idx[i]] = 1
    s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)

    with torch.no_grad():
        probits_batch = network(x.to(args.device))[:, 1:] 
        probits_list.append(probits_batch)
                
    probits = torch.cat(probits_list).detach().cpu().numpy()
    labels = s.argmax(1)

##################################################

import os
path_results = 'results/benchmark/'
path_figs = 'dx_figs/benchmark/'
os.makedirs(path_results, exist_ok=True)
os.makedirs(path_figs, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--num_domains', '-n', default=4, type=int)
parser.add_argument('--seed', '-s', default=0, type=int)
args = parser.parse_args()

dat = pd.read_feather('data/insight/static_new.feather')

# Define the path to the dataset folder
data_path = "/path/to/dataset"

print(dat.shape)

train_val_dat, test_dat = train_test_split(dat, stratify=dat['SITE']+dat['CVD'].astype(str), test_size=0.2, random_state=args.seed)

domain_color_map = {
    'A05': 'orange',
    'A07': 'navy',
    'B03': 'darkgreen',
    'C02': 'slateblue',
    'C05': 'crimson'
}
domain_color_map = {
    '0': 'orange',
    '1': 'navy',
    '2': 'darkgreen',
    '3': 'slateblue',
    '4': 'crimson'
}

covariate_list = train_val_dat.columns.drop(['SSID', 'SITE', 'CVD'])
num_prefix = ['AGE', 'VITAL', 'LAB', 'ENC', 'FLWUP']
num_cols = [c for c in covariate_list if any(c.startswith(prefix) for prefix in num_prefix)]
cat_cols = [c for c in covariate_list if not any(c.startswith(prefix) for prefix in num_prefix)]
sites = test_dat['SITE'].unique().tolist()

simple_stack = []
train_domains = []
for train_sites in combinations(sites, args.num_domains):
    train_sites = sorted(train_sites)
    train_dat, val_dat = train_test_split(train_val_dat, stratify=train_val_dat['SITE'] + train_val_dat['CVD'].astype(str), test_size=0.2, random_state=args.seed)

    train_idx = [s in train_sites for s in train_dat['SITE'].tolist()]
    valid_idx = [s in train_sites for s in val_dat['SITE'].tolist()]

    tr_s = train_dat[train_idx]['SITE']
    tr_x = train_dat[train_idx][num_cols].values.astype(float)
    tr_y = train_dat[train_idx]['CVD'].values.astype(float)

    val_s = val_dat[valid_idx]['SITE']
    val_x = val_dat[valid_idx][num_cols].values.astype(float)
    val_y = val_dat[valid_idx]['CVD'].values.astype(float)

    num_imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    tr_x = num_imputer.fit_transform(tr_x)
    tr_x = scaler.fit_transform(tr_x)
    
    ''' PRETRAINED WEIGHT '''
    ''' IMAGE SCALER '''
    val_x = num_imputer.transform(val_x)
    val_x = scaler.transform(val_x)

    tr_x = np.column_stack([tr_x, train_dat[train_idx][cat_cols].values.astype(float)])
    val_x = np.column_stack([val_x, val_dat[valid_idx][cat_cols].values.astype(float)])

    max_epoch = 150

    network = SeparatedHeckmanNetwork([tr_x.shape[1], 128, 64, 32, 1], [tr_x.shape[1], 64, 32, 16, args.num_domains], dropout=0.5, batchnorm=True, activation='ReLU') #TODO: follow the ac
    # g_layers[-1]
    # g_layers = [tr_x.shape[1], 64, 32, 16, args.num_domains]
    network_cnn = SeparatedHeckmanNetworkCNN(args)
    optimizer = partial(torch.optim.SGD, lr=1e-2, weight_decay=1e-2)
    #scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=50, eta_min=1e-6)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 75, 100, 125], gamma=.1)
    model = HeckmanDGBinaryClassifier(network, optimizer, scheduler, config={'max_epoch': max_epoch})
    model.fit(
        {'train_x': tr_x,
         'train_y': tr_y,
         'train_s': pd.get_dummies(tr_s).values,
         'valid_x': val_x,
         'valid_y': val_y,
         'valid_s': pd.get_dummies(val_s).values})

    domain_names = pd.get_dummies(val_s).columns
    plt.figure(figsize=(6, 9))
    plt.subplot(311)
    plt.title(f'loss curves')
    plt.plot(model.train_loss_traj, label='train loss', color='royalblue')
    plt.plot(model.valid_loss_traj, label='valid loss', color='limegreen')
    plt.plot(model.best_epoch, model.best_valid_loss, marker='v', color='forestgreen', label='best valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_train_loss, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.legend()

    plt.subplot(312)
    plt.title(f'auroc curves')
    plt.plot(model.train_auc_traj, label='train auroc', color='royalblue')
    plt.plot(model.valid_auc_traj, label='valid auroc', color='limegreen')
    plt.plot(model.best_epoch, model.best_valid_auc, marker='v', color='forestgreen', label='best valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_train_auc, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.legend()

    plt.subplot(313)
    plt.title(f'rho trajectory')
    for s in range(args.num_domains):
        plt.plot(np.array(model.rho_traj)[:, s],
                 label=domain_names[s],
                 color=domain_color_map[domain_names[s]],
                 marker='x', alpha=.5)
    plt.vlines(model.best_epoch, -0.99, 0.99, linestyles=':', color='gray')
    plt.legend()
    plt.ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(f"dx_figs/heckmanDG_{'+'.join(train_sites)}_{args.seed}.pdf")
    plt.close()

    probits, labels = model.get_selection_probit({'train_x': tr_x, 'train_s': pd.get_dummies(tr_s).values})

    plt.figure(figsize=(4, 2*args.num_domains))
    for s in range(args.num_domains):
        plt.subplot(args.num_domains, 1, s+1)
        plt.title(f'Selection Model for {domain_names[s]}')
        for ss in range(args.num_domains):
            plt.hist(probits[labels == ss, s],
                     label=domain_names[ss],
                     color=domain_color_map[domain_names[ss]],
                     alpha=.5, bins=100)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"dx_figs/heckmanDG_{'+'.join(train_sites)}_{args.seed}_probits.pdf")
    plt.close()

    all, internal, external = [], [], []
    for site in sites:
        te_x = test_dat[test_dat['SITE'] == site][num_cols].values.astype(float)
        te_y = test_dat[test_dat['SITE'] == site]['CVD'].values.astype(float)
        te_x = num_imputer.transform(te_x)
        te_x = scaler.transform(te_x)
        te_x = np.column_stack([te_x, test_dat[test_dat['SITE'] == site][cat_cols].values.astype(float)])
        score = roc_auc_score(te_y, model.predict_proba(te_x))
        if site in train_sites:
            internal.append(score)
        else:
            external.append(score)
        all.append(score)

    simple_stack.append(all + [np.mean(internal), np.mean(external), np.mean(all)])

    train_domains.append('+'.join(train_sites))
    print(pd.DataFrame(simple_stack, index=train_domains, columns=sites + ['Internal', 'External', 'All']))

pd.DataFrame(simple_stack, index=train_domains, columns=sites + ['Internal', 'External', 'All']).to_csv(f'results/heckmanDG_{args.num_domains}_{args.seed}.csv')

# for batch in train_loader:
    
#     x_batch = batch['x']  # Get the 'x' tensor from the batch dictionary
#     y_batch = batch['y']  # Get the 'y' tensor from the batch dictionary
#     s_batch_ = batch['domain']  # Get the 'y' tensor from the batch dictionary
    
#     s_batch_ohe = torch.nn.functional.one_hot(s_batch_)#, num_classes=args.num_domains
#     non_zero_cols = torch.any(s_batch_ohe, dim=0)
#     # Index the tensor using the non-zero columns
#     s_batch = s_batch_ohe[:, non_zero_cols]
#     print(x_batch)
#     print(y_batch)
#     print(s_batch)
#     # Print the shapes of the 'x' and 'y' tensors
#     print('x_batch shape:', x_batch.shape)  # (batch_size, channels, w, h)
#     print('y_batch shape:', y_batch.shape)  # (batch_size,)
#     print('s_batch shape:', s_batch.shape)  # (batch_size, n_domains)

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(s_.cpu())
# onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# s = torch.Tensor(onehot_encoded).to(args.device).to(torch.float32) # (B, *)
        


##### TODO
import typing

# 4-1. Correlation
if defaults.loss_type == 'multiclass':
    J: int = NetworkInitializer._out_features[args.data][0]  # number of outcome classes
    K: int = len(train_domains)                              # number of training domains
    _rho = torch.randn(K, J + 1, device=device, requires_grad=True)
    _rho = nn.Parameter(_rho)
else:
    K: int = len(train_domains)
    _rho = torch.zeros(K, device=args.device, requires_grad=True)
    _rho = nn.Parameter(data=_rho)

# 4-2. Sigma (for regression only)
if defaults.loss_type == 'regression':
    sigma = nn.Parameter(torch.ones(1, device=args.device), requires_grad=True)

# 5. temperature
self.temperature: typing.Union[torch.FloatTensor, float] = 1.0

_layers = []
for d, (units_in, units_out) in enumerate(zip(f_layers, f_layers[1:])):
    _layers.append(nn.Linear(units_in, units_out, bias=bias))
    if d < len(f_layers) - 2:
        if batchnorm:
            _layers.append(nn.BatchNorm1d(units_out))
        _layers.append(activation())
        if dropout:
            _layers.append(nn.Dropout(dropout))

self.f_layers = nn.Sequential(*_layers)

_layers = []
for d, (units_in, units_out) in enumerate(zip(g_layers, g_layers[1:])):
    _layers.append(nn.Linear(units_in, units_out, bias=bias))
    if d < len(g_layers) - 2:
        if batchnorm:
            _layers.append(nn.BatchNorm1d(units_out))
        _layers.append(activation())
        if dropout:
            _layers.append(nn.Dropout(dropout))

self.g_layers = nn.Sequential(*_layers)
'''



'''
if data_name == 'camelyon17':
    data_type = 'image'
    args = args_cameloyn17(args, experiment_name) # data-specific arguments 
    # DataDefaults: has data-specific hyperparameters
    defaults = DataDefaults[args.data]() 
    args.root = defaults.root#: str = './data/benchmark/wilds/camelyon17_v1.0'
    args.train_domains = defaults.train_domains#: typing.List[int] = [0, 3, 4]  # [0, 3, 4]
    args.validation_domains = defaults.validation_domains#: typing.List[int] = [1]   # [1]
    args.test_domains = defaults.test_domains#: typing.List[int] = [2]         # [2]
    args.load_data_in_memory = defaults.load_data_in_memory#: int = 0
    args.model_selection = defaults.model_selection#: str = 'ood'
    args.model_selection_metric = defaults.model_selection_metric#: str = 'accuracy' #TODO: ADD ECE
    args.num_classes = defaults.num_classes#: int = 2
    args.loss_type = defaults.loss_type#: str = 'binary'
    args.num_domains = len(defaults.train_domains)
elif data_name == 'poverty':
    data_type = 'image'
    args = args_poverty(args, experiment_name) # data-specific arguments 
    defaults = DataDefaults[args.data]() 
    args.root = defaults.root#: str = './data/benchmark/wilds/poverty_v1.1'
    args.fold = defaults.fold#: str = 'A'  # [A, B, C, D, E]
    args.use_area = defaults.use_area#: bool = False
    args.load_data_in_memory = defaults.load_data_in_memory#: int = 0
    args.model_selection = defaults.model_selection#: str = 'ood'
    args.model_selection_metric = defaults.model_selection_metric#: str = 'pearson'
    args.num_classes = defaults.num_classes#: int = None  # type: ignore
    args.loss_type = defaults.loss_type#: str = 'regression'
elif data_name == 'rxrx1':
    data_type = 'image'
    args = args_rxrx1(args, experiment_name) # data-specific arguments 
    defaults = DataDefaults[args.data]() 
    args.root = defaults.root #: str = './data/benchmark/wilds/rxrx1_v1.0'
    args.load_data_in_memory = defaults.load_data_in_memory #: int = 0  # 12 or 0
    args.model_selection = defaults.model_selection #: str = 'ood'
    args.model_selection_metric = defaults.model_selection_metric #: str = 'accuracy'    
    args.num_classes = defaults.num_classes #: int = 1139
    args.loss_type = defaults.loss_type #: str = 'multiclass'
elif data_name == 'iwildcam':
    data_type = 'image'
    args = args_iwildcam(args, experiment_name) # data-specific arguments 
    defaults = DataDefaults[args.data]() 
    args.root = defaults.root#: str = './data/benchmark/wilds/iwildcam_v2.0'
    args.model_selection = defaults.model_selection#: str = 'ood'
    args.model_selection_metric = defaults.model_selection_metric#: str = 'f1'  # macro it is
    args.num_classes = defaults.num_classes#: int = 182
    args.loss_type = defaults.loss_type#: str = 'multiclass'
else:
    print("choose the data_name among ('camelyon17', 'camelyon17_ece', 'poverty', 'rxrx1', 'iwildcam')")
    
'''