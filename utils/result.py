import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def prediction(data_loader, model, args):
    # data_loader = train_loader
    
    res = []
    for b, batch in enumerate(data_loader):
        print(f'loader {b} batch / {len(data_loader)}')
        y_true = batch['y']
        if args.loss_type == 'binary':
            y_pred = model.predict_proba(batch)
            threshold = 0.5
            y_pred_cl = (y_pred>threshold).astype(int)
            # auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred_cl)
            f1_mac = f1_score(y_true, y_pred_cl, average='macro').round(3)
            # res.append([auc, acc, f1_mac])
            res.append([acc, f1_mac])
            
        elif args.loss_type == 'regression':
            y_pred = model.predict(batch)
            mse = mean_squared_error(y_true, y_pred).round(3)
            mae = mean_absolute_error(y_true, y_pred).round(3)
            pearson = pearsonr(y_true, y_pred)[0].round(3)
            res.append([mse, mae, pearson])
            
        elif args.loss_type == 'multiclass':
            y_pred = model.predict_proba(batch)
            idx = y_pred.argmax(axis=1)
            y_pred_class = torch.zeros(len(y_pred), args.num_classes, dtype=torch.int, device=args.device)
            for i in range(len(y_pred_class)):
                y_pred_class[i, idx[i]] = 1
            acc = accuracy_score(y_true, y_pred_class)
            f1_mac = f1_score(y_true, y_pred_class, average='macro', zero_division=0).round(3) ###
            res.append([acc, f1_mac])
        else:
            print("choose binary, regression, or multiclass")
        
    return res
 
def plots_loss(model, args, domain_color_map, path: str):
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
    plt.plot(model.valid_loss_traj, label='valid auroc', color='limegreen')
    plt.plot(model.best_epoch, model.best_valid_auc, marker='v', color='forestgreen', label='best valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_train_auc, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.legend()

    plt.subplot(313)
    plt.title(f'rho trajectory')
    for s in range(args.num_domains):
        plt.plot(np.array(model.rho_traj)[:, s],
                    label=args.train_domains[s],
                    color=domain_color_map[args.train_domains[s]],
                    marker='x', alpha=.5)
    plt.vlines(model.best_epoch, -0.99, 0.99, linestyles=':', color='gray')
    plt.legend()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plots_loss_id_ood(model, args, path: str):
    plt.figure(figsize=(6, 9))
    plt.subplot(111)
    plt.title(f'loss curves')
    plt.plot(model.train_loss_traj, label='train loss', color='red')
    plt.plot(model.id_valid_loss_traj, label='valid loss', color='green')
    plt.plot(model.ood_valid_loss_traj, label='valid loss', color='blue')
    plt.plot(model.best_epoch, model.best_train_loss, marker='^', color='navy', label='best train loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_id_valid_loss, marker='v', color='navy', label='best id valid loss', alpha=.5)
    plt.plot(model.best_epoch, model.best_ood_valid_loss, marker='v', color='navy', label='best ood valid loss', alpha=.5)
    plt.legend()

    plt.subplot(112)
    plt.title(f'rho trajectory')
    for s in range(args.num_domains):
        plt.plot(np.array(model.rho_traj)[:, s],
                    label=args.train_domains[s],
                    # color=domain_color_map[args.train_domains[s]],
                    marker='x', alpha=.5)
    plt.vlines(model.best_epoch, -0.99, 0.99, linestyles=':', color='gray')
    plt.legend()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plots_probit(probits, labels, args, path:str):
    plt.figure(figsize=(4, 2*args.num_domains))
    for s in range(args.num_domains):
        print(f'{s} th domain: {args.train_domains[s]}')
        plt.subplot(args.num_domains, 1, s+1)
        plt.title(f'Selection Model for {args.train_domains[s]}')
        for ss in range(args.num_domains):
            print(f'{ss} th domain {args.train_domains[ss]} in {args.train_domains[s]} domain  probits')
            probits_ = probits[np.where(labels == ss), s].reshape(-1)
            plt.hist(probits_,
                    label=args.train_domains[ss],
                    # color=domain_color_map[args.train_domains[ss]],
                    alpha=.5, bins=3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    
# def plots_probit(probits, labels, args, domain_color_map, path:str):
    
#     plt.figure(figsize=(4, 2*args.num_domains))
#     for s in range(args.num_domains):
#         print(f'{s} th domain: {args.train_domains[s]}')
#         plt.subplot(args.num_domains, 1, s+1)
#         plt.title(f'Selection Model for {args.train_domains[s]}')
#         for ss in range(args.num_domains):
#             print(f'{ss} th domain {args.train_domains[ss]} in {args.train_domains[s]} domain  probits')
#             probits_ = probits[np.where(labels == ss), s].reshape(-1)
#             plt.hist(probits_,
#                     label=args.train_domains[ss],
#                     color=domain_color_map[args.train_domains[ss]],
#                     alpha=.5, bins=3)
#         plt.legend()

#     plt.tight_layout()
#     plt.savefig(path)
#     plt.close()
