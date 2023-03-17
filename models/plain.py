
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from tqdm import tqdm

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))


class ProbitBinaryClassifier:
    def __init__(self, network, optimizer, scheduler, config=dict()):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = {
            'device': 'cuda',
            'max_epoch': 100,
            'batch_size': 500
        }

        self.config.update(config)
        self.network = self.network.to(self.config['device'])

    def fit(self, data):
        dataloaders = dict()
        for split in ['train', 'valid']:
            dataloaders[split] = DataLoader(
                TensorDataset(
                    torch.FloatTensor(data[f'{split}_x']),
                    torch.FloatTensor(data[f'{split}_y'])
                ),
                shuffle=(split == 'train'), batch_size=self.config['batch_size'], drop_last=True
            )

        opt = self.optimizer(params=self.network.parameters())
        sch = self.scheduler(opt) if self.scheduler else None

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        best_model, best_loss, best_acc = deepcopy(self.network.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)
        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.network.train()
            train_loss = 0.
            train_pred, train_true = [], []
            for batch in dataloaders['train']:
                batch = [b.to(self.config['device']) for b in batch]
                opt.zero_grad()

                #batch[0][:50] += torch.randn_like(batch[0][:50], device=batch[0].device)*0.1

                batch_probit = self.network(batch[0]).squeeze()
                batch_prob = normal.cdf(batch_probit)

                loss = -(batch[1] * safe_log(batch_prob) + (1-batch[1])*safe_log(1-batch_prob)).mean()

                loss.backward()
                opt.step()

                train_loss += loss.item() / len(dataloaders['train'])
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(batch[1].detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.concatenate(train_true),
                    np.concatenate(train_pred)))

            if sch:
                sch.step()

            self.network.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                for batch in dataloaders['valid']:
                    batch = [b.to(self.config['device']) for b in batch]
                    batch_probit = self.network(batch[0]).squeeze()
                    batch_prob = normal.cdf(batch_probit)

                    loss = -(batch[1] * safe_log(batch_prob) + (1-batch[1])*safe_log(1-batch_prob)).mean()
                    valid_loss += loss.item() / len(dataloaders['valid'])
                    valid_pred.append(batch_prob.detach().cpu().numpy())
                    valid_true.append(batch[1].detach().cpu().numpy())

            if valid_loss < best_loss:
                best_model = deepcopy(self.network.state_dict())
                best_loss = valid_loss
                best_epoch = epoch

            valid_loss_traj.append(valid_loss)
            valid_auc_traj.append(
                roc_auc_score(
                    np.concatenate(valid_true),
                    np.concatenate(valid_pred)))

            desc = f'[{epoch + 1:03d}/{self.config["max_epoch"]:03d}] '
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            desc += f'AUROC: {train_auc_traj[-1]:.4f} '
            desc += f'| valid | Loss: {valid_loss_traj[-1]:.4f} '
            desc += f'AUROC: {valid_auc_traj[-1]:.4f} '
            pbar.set_description(desc)

        self.network.load_state_dict(best_model)
        self.train_loss_traj = train_loss_traj
        self.valid_loss_traj = valid_loss_traj
        self.train_auc_traj = train_auc_traj
        self.valid_auc_traj = valid_auc_traj
        self.best_train_loss = train_loss_traj[best_epoch]
        self.best_valid_loss = valid_loss_traj[best_epoch]
        self.best_train_auc = train_auc_traj[best_epoch]
        self.best_valid_auc = valid_auc_traj[best_epoch]
        self.best_epoch = best_epoch

    def predict_proba(self, X):
        dataloader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            shuffle=False, batch_size=self.config['batch_size'], drop_last=False
        )

        normal = torch.distributions.normal.Normal(0, 1)
        with torch.no_grad():
            probs = torch.cat(
                [normal.cdf(self.network(batch_x.to(self.config['device'])).squeeze()) for (batch_x, ) in dataloader]
            ).detach().cpu().numpy()

        return probs
