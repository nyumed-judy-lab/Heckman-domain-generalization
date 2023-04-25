#%% Modules
import os
import torch
import pickle
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

from utils.argparser import DatasetImporter, parse_arguments
from utils.argparser import args_insight
from utils.argparser import args_cameloyn17, args_poverty, args_iwildcam, args_rxrx1
from utils.dataloader import dataloaders, sub_dataloaders
from utils.seed import fix_random_seed

os.makedirs('./results', exist_ok=True)
os.makedirs('./results/plots/', exist_ok=True)
os.makedirs('./results/prediction/', exist_ok=True)
os.makedirs('./results/models/', exist_ok=True)


#%% 1. Experiment Settings
# from utils_datasets.defaults import DataDefaults
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

data_name = 'iwildcam'  #'rxrx1', 'camelyon17', 'iwildcam', 'poverty', 'iwildcam'
data_name = 'rxrx1'  #'rxrx1'
experiment_name = 'Heckman DG'
args = parse_arguments(experiment_name) # basic arguments for image data
args = data_argument(args, data_name)

if False:
    args.device = 'cpu'

########################################
seeds = [0,1,2,3,4,5,6,7,8,9]
id_proportion = [0.0, 0.5, 1.0]
model_selection_list = ['loss', 'metric']
epochs = []
# args.batch_size = 500
# args.epochs = 2
# args.device='cuda:2'
for seed in seeds:
    for w in id_proportion:
        for selection in model_selection_list:
            args.seed = seed # 0.5 # id_proportion            
            args.w = w # 0.5 # id_proportion
            args.model_selection = selection #'loss'#, 'metric'
            
            result_name = f'{args.data}_{args.seed}_w_{args.w}_selection_{args.model_selection}'
            fix_random_seed(args.seed)
            print(result_name)
            '''
            args.seed = 0  
            args.w =  0.0 # id_proportion
            args.model_selection = 'loss' # 'metric'
            '''
            ########################################            
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
            if False:
                result_name = result_name+'_example'
                train_loader, id_valid_loader, ood_valid_loader, test_loader = sub_dataloaders(train_loader, id_valid_loader, ood_valid_loader, test_loader)
            if False:
                batch = next(iter(train_loader))
                batch_id = next(iter(id_valid_loader))
                batch_ood = next(iter(ood_valid_loader))
                # save the batch using pickle
                with open(f'./data/examples/batch_{args.data}.pkl', 'wb') as f:
                    pickle.dump(batch, f)
                with open(f'./data/examples/batch_id_{args.data}.pkl', 'wb') as f:
                    pickle.dump(batch_id, f)
                with open(f'./data/examples/batch_ood_{args.data}.pkl', 'wb') as f:
                    pickle.dump(batch_ood, f)
            if False:
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

            #%% HeckmanDG
            import tqdm
            from copy import deepcopy
            import torch.nn.functional as F
            from utils_datasets.transforms import InputTransforms
            from networks import HeckmanDNN, HeckmanCNN 
            from models.heckmandg_regression import HeckmanDG_CNN_Regressor
            from models.heckmandg_binaryclass import HeckmanDG_CNN_BinaryClassifier
            from models.heckmandg_multiclass import HeckmanDG_CNN_MultiClassifier
            optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
            scheduler = None
            if args.loss_type == 'binary':
                model = HeckmanDG_CNN_BinaryClassifier(args, network, optimizer)
            elif args.loss_type == 'regression':
                model = HeckmanDG_CNN_Regressor(args, network, optimizer)
            elif args.loss_type == 'multiclass':
                model = HeckmanDG_CNN_MultiClassifier(args, network, optimizer)
            model.fit(train_loader, id_valid_loader, ood_valid_loader)
            torch.save(model.network.state_dict(), f'./results/models/{result_name}_model.pth')

            if False:
                # load the model
                network = HeckmanCNN(args)
                optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay)
                if args.loss_type == 'binary':
                    model = HeckmanDG_CNN_BinaryClassifier(args, network, optimizer)
                elif args.loss_type == 'regression':
                    model = HeckmanDG_CNN_Regressor(args, network, optimizer)
                elif args.loss_type == 'multiclass':
                    model = HeckmanDG_CNN_MultiClassifier(args, network, optimizer)
                # Load the saved model state dictionary
                state_dict = torch.load(f'./results/{result_name}_model.pth')
                # Load the saved state dictionary into the model
                model.network.load_state_dict(state_dict)
                
            #%% 4. Results Analysis
            from utils.result import prediction, plots_loss_id_ood, plots_loss, plots_probit
            # prediction results
            res_tr = prediction(train_loader, model, args)
            res_vl_id = prediction(id_valid_loader, model, args)
            res_vl_ood = prediction(ood_valid_loader, model, args)
            res_ts = prediction(test_loader, model, args)

            res_tr_mean = pd.DataFrame(res_tr).mean().round(3)
            res_id_vl_mean = pd.DataFrame(res_vl_id).mean().round(3)
            res_ood_vl_mean = pd.DataFrame(res_vl_ood).mean().round(3)
            res_ts_mean = pd.DataFrame(res_ts).mean().round(3)

            results = np.concatenate((np.array([f'{args.data}']), 
                                      res_tr_mean.values, 
                                      res_id_vl_mean.values, 
                                      res_ood_vl_mean.values, 
                                      res_ts_mean.values))
            results = pd.DataFrame([results])
            ##### Binary Classification Columns columns
            results.columns = ['data', 
                            # 'train_auc', 
                            'train_acc', 'train_f1', 
                            # 'valid_id_auc', 
                            'valid_id_acc', 'valid_id_f1',
                            # 'valid_ood_auc', 
                            'valid_ood_acc', 'valid_ood_f1', 
                            # 'test_auc', 
                            'test_f1', 'test_acc']
            results.to_csv(f'./results/prediction/prediction_{result_name}.csv')
            print(results)

            # plots: loss, probits
            try:
                probits, labels = model.get_selection_probit(train_loader)
                plots_loss_id_ood(model, args, path=f"./results/plots/loss_{result_name}.pdf")
                plots_probit(probits, labels, args, path=f"./results/plots/probits_{result_name}.pdf")
            except ValueError:
                print('no plots')
                
            
