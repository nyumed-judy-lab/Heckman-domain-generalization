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

data_name = 'insight' #'insight', 'camelyon17', 'iwildcam', 'iwildcam', 'iwildcam', 'iwildcam'
experiment_name = 'Heckman DG'
args = args_insight()
dat = pd.read_feather(args.root)

#%% 2. Data Preparation
covariate_list = dat.columns.drop(['SSID', 'SITE', 'CVD'])
num_prefix = ['AGE', 'LAB', 'ENC', 'FLWUP', 
              'VITAL_HT', 'VITAL_WT', 
              'VITAL_DIASTOLIC', 'VITAL_SYSTOLIC', 
              'VITAL_BMI'] #'VITAL', 
num_cols = [c for c in covariate_list if any(c.startswith(prefix) for prefix in num_prefix)]
cat_cols = [c for c in covariate_list if not any(c.startswith(prefix) for prefix in num_prefix)]
domains = dat['SITE'].unique().tolist()

#%% 3. HeckmanDG
from networks import HeckmanDNN, HeckmanCNN 
from models import HeckmanDGBinaryClassifier, HeckmanDGBinaryClassifierCNN
from utils_datasets.transforms import InputTransforms
from utils.preprocessing import DatasetImporter_tabular

domain_color_map = {
    'A05': 'orange',
    'A07': 'navy',
    'B03': 'darkgreen',
    'C02': 'slateblue',
    'C05': 'crimson'
}

train_val_dat, test_dat = train_test_split(dat, stratify=dat['SITE']+dat['CVD'].astype(str), test_size=args.test_size, random_state=args.seed)

res = []
train_domains = []
for train_sites in combinations(domains, args.num_domains):
    args.train_domains = train_sites
    
    tr_x, tr_s, tr_y, val_x, val_s, val_y, num_imputer, scaler = DatasetImporter_tabular(train_val_dat, args, num_cols, cat_cols)
    
    network = HeckmanDNN([tr_x.shape[1], 128, 64, 32, 1], [tr_x.shape[1], 64, 32, 16, args.num_domains], dropout=0.5, batchnorm=True, activation='ReLU')
    optimizer = partial(torch.optim.SGD, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 75, 100, 125], gamma=.1)
    model = HeckmanDGBinaryClassifier(network, optimizer, scheduler, config={'max_epoch': args.epochs})
    
    model.fit(
        {'train_x': tr_x,
         'train_y': tr_y,
         'train_s': pd.get_dummies(tr_s).values,
         'valid_x': val_x,
         'valid_y': val_y,
         'valid_s': pd.get_dummies(val_s).values})
    # save the model
    torch.save(model.network.state_dict(), f'./results/{args.data}_{args.seed}_model.pth')
    if False:
        # load the model
        model = HeckmanDGBinaryClassifier()
        # Load the saved model state dictionary
        state_dict = torch.load(f'./results/{args.data}_{args.seed}_model.pth')
        # Load the saved state dictionary into the model
        model.load_state_dict(state_dict)

    #%% 4. Results Analysis
    from utils.result import prediction, plots_loss, plots_probit
    
    plots_loss(model, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_{train_sites}_loss.pdf")
    probits, labels = model.get_selection_probit({'train_x': tr_x, 'train_s': pd.get_dummies(tr_s).values})
    plots_probit(probits, labels, args, domain_color_map, path=f"./results/plots/HeckmanDG_{args.data}_{train_sites}_probits.pdf")

    all, internal, external = [], [], []
    for site in domains:
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

    res.append(all + [np.mean(internal), np.mean(external), np.mean(all)])

    train_domains.append('+'.join(train_sites))
    print(pd.DataFrame(res, index=train_domains, columns=domains + ['Internal', 'External', 'All']))

results = pd.DataFrame(res, index=train_domains, columns=domains + ['Internal', 'External', 'All'])
results.to_csv(f'./results/heckmanDG_{args.data}_{args.num_domains}_{args.seed}.csv')
