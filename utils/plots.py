import numpy as np
import matplotlib.pyplot as plt

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
    plt.plot(model.valid_auc_traj, label='valid auroc', color='limegreen')
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

def plots_probit(probits, labels, args, domain_color_map, path:str):
    
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
                    color=domain_color_map[args.train_domains[ss]],
                    alpha=.5, bins=3)
        plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
