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
from utils.argparser import args_insight
from utils.argparser import args_cameloyn17, args_poverty, args_iwildcam, args_rxrx1
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

# cheack data availability
for data_name in data_list:
    # data_name = 'insight', 'rxrx1'  #'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
    # data_name = 'poverty'  #'insight', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
    experiment_name = 'Heckman DG'
    args = parse_arguments(experiment_name) # basic arguments for image data
    
    if data_name == 'poverty':
        for fold in poverty_folds: 
            args = data_argument(args, data_name, fold=fold)
            dataset = DatasetImporter(args)
            print('data_name', data_name)
            print(dataset)
            if args.num_domains is not None:
                print(args.num_domains)
    else:
        args = data_argument(args, data_name)
        dataset = DatasetImporter(args)
        print('data_name', data_name)
        print(dataset)
        if args.num_domains is not None:
            print(args.num_domains)

if False:
    # args.data = 'poverty' 
    import torch
    import typing
    from utils_datasets.wilds_ import SingleCamelyon, WildsCamelyonDataset
    from utils_datasets.wilds_ import SinglePovertyMap, WildsPovertyMapDataset
    from utils_datasets.wilds_ import SingleIWildCam, WildsIWildCamDataset
    from utils_datasets.wilds_ import SingleRxRx1, WildsRxRx1Dataset

    import pickle
    with open('./data/benchmark/poverty_v1.1/dhs_incountry_folds.pkl', 'rb') as f:
        dhs_incountry_folds = pickle.load(f)
    dhs_incountry_folds[fold]['train']
    dhs_incountry_folds[fold]['val']
    dhs_incountry_folds[fold]['test']
    dhs_incountry_folds_df = pd.DataFrame(dhs_incountry_folds)
    dhs_incountry_folds_df.to_csv('./results/povertymap_folds.csv')

    import pandas as pd
    dhs_metadata = pd.read_csv('./data/benchmark/poverty_v1.1/dhs_metadata.csv')
    import os
    images = os.listdir('./data/benchmark/poverty_v1.1/images/')
    len(images)


    for fold in poverty_folds:
        # fold = 'E'
        if fold == 'A':
            train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['train']
            validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_val']
            test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_test']
        elif fold == 'B':
            train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['train']
            validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['ood_val']
            test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17B['ood_test']
        elif fold == 'C':
            train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['train']
            validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['ood_val']
            test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17C['ood_test']
        elif fold == 'D':
            train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['train']
            validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['ood_val']
            test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17D['ood_test']
        elif fold == 'E':
            train_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['train']
            validation_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['ood_val']
            test_countries: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17E['ood_test']
        else:
            print("choose in ['A','B','C','D','E']")

        dataset = WildsPovertyMapDataset(
            args = args,
            root=args.root,
            train_domains = train_countries,
            validation_domains = validation_countries,
            test_domains=test_countries,
        )
        setattr(args, 'train_domains', dataset.train_domains)
        setattr(args, 'validation_domains', dataset.validation_domains)
        setattr(args, 'test_domains', dataset.test_domains)
        setattr(args, 'num_domains', len(dataset.train_domains))
    print(fold, 'args.num_domains', args.num_domains, dataset)


