# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 15:31:36 2023

@author: HR
"""

import os
os.makedirs('results', exist_ok=True)
os.makedirs('dx_figs', exist_ok=True)

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from functools import partial
from itertools import combinations
import argparse
from models import ProbitBinaryClassifier
from networks import BasicNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--num_domains', '-n', default=4, type=int)
parser.add_argument('--seed', '-s', default=0, type=int)
args = parser.parse_args()

dat = pd.read_feather('data/static.feather')

print(dat.shape)

simple_repeat = []

train_val_dat, test_dat = train_test_split(dat, stratify=dat['SITE']+dat['CVD'].astype(str), test_size=0.2, random_state=args.seed)

domain_color_map = {
    'A05': 'orange',
    'A07': 'navy',
    'B03': 'darkgreen',
    'C02': 'slateblue',
    'C05': 'crimson'
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
    val_x = num_imputer.transform(val_x)
    val_x = scaler.transform(val_x)

    tr_x = np.column_stack([tr_x, train_dat[train_idx][cat_cols].values.astype(float)])
    val_x = np.column_stack([val_x, val_dat[valid_idx][cat_cols].values.astype(float)])

    max_epoch = 100
    network = BasicNetwork([tr_x.shape[1], 128, 64, 32, 1], dropout=0.1, batchnorm=True, activation='ReLU')
    optimizer = partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-2)
    #scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=50, eta_min=1e-6)
    scheduler = partial(torch.optim.lr_scheduler.MultiStepLR, milestones=[50, 75], gamma=.1)
    model = ProbitBinaryClassifier(network, optimizer, scheduler, config={'max_epoch': max_epoch})
    model.fit({'train_x': tr_x, 'train_y': tr_y, 'valid_x': val_x, 'valid_y': val_y})

    plt.figure(figsize=(6, 6))
    plt.subplot(211)
    plt.title(f'loss curves')
    plt.plot(model.train_loss_traj, label='train loss', color='royalblue')
    plt.plot(model.valid_loss_traj, label='valid loss', color='limegreen')
    plt.plot(model.best_epoch, model.best_valid_loss, marker='v', color='forestgreen', label='best valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_train_loss, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.legend()

    plt.subplot(212)
    plt.title(f'auroc curves')
    plt.plot(model.train_auc_traj, label='train auroc', color='royalblue')
    plt.plot(model.valid_auc_traj, label='valid auroc', color='limegreen')
    plt.plot(model.best_epoch, model.best_valid_auc, marker='v', color='forestgreen', label='best valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_train_auc, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"dx_figs/ERM_{'+'.join(train_sites)}_{args.seed}.pdf")

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

pd.DataFrame(simple_stack, index=train_domains, columns=sites + ['Internal', 'External', 'All']).to_csv(f'results/ERM_{args.num_domains}_{args.seed}.csv')

