
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from tqdm import tqdm

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))


def _linspace_with_grads(start: torch.Tensor, stop: torch.Tensor, steps: int, device: str = 'cpu'):
    """
    Creates a 1-dimensional grid while preserving the gradients.
    Reference:
        https://github.com/esa/torchquad/blob/4be241e8462949abcc8f1ace48d7f8e5ee7dc136/torchquad/integration/utils.py#L7
    """
    grid = torch.linspace(0, 1-1e-6, steps, device=device)  # Create a 0 to 1 spaced grid
    grid *= stop - start                               # Scale to desired range, thus keeping gradients
    grid += start
    return grid

def bivariate_normal_cdf(a: torch.Tensor, b: torch.Tensor, rho: torch.FloatTensor, steps: int = 100):
    """
    Approximation of standard bivariate normal cdf using the trapezoid rule.
    The decomposition is based on:
        Drezner, Z., & Wesolowsky, G. O. (1990).
        On the computation of the bivariate normal integral.
        Journal of Statistical Computation and Simulation, 35(1-2), 101-107.
    Arguments:
        a: 1D tensor of shape (B, )
        b: 1D tensor of shape (B, )
        rho:
    """
    device = a.device
    a, b = a.view(-1, 1), b.view(-1, 1)  # for proper broadcasting with x
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    x = _linspace_with_grads(start=0, stop=rho, steps=steps,
                             device=device).unsqueeze(0).expand(a.shape[0], -1)  # (B, steps)
    # x = _linspace_with_grads(start=0, stop=rho, steps=steps).repeat(a.shape[0], 1)
    y = 1 / torch.sqrt(1 - torch.pow(x, 2)) * torch.exp(
        - (torch.pow(a, 2) + torch.pow(b, 2) + 2 * a * b * x) / (2 * (1 - torch.pow(x, 2)))
        )

    return _normal.cdf(a.squeeze()) * _normal.cdf(b.squeeze()) + \
        (1 / (2 * torch.pi)) * torch.trapezoid(y=y, x=x)


class LinearHeckmanDG(nn.Module):
    def __init__(self, n_covariates, n_domains):
        super(LinearHeckmanDG, self).__init__()
        self.outcome_model = nn.Linear(n_covariates, 1)
        self.selection_model = nn.Linear(n_covariates, n_domains)
        self.rho = nn.Parameter(torch.zeros(n_domains, dtype=torch.float), requires_grad=True)
        self.register_parameter(name='rho', param=self.rho)

    def forward(self, x):
        return torch.cat([self.outcome_model(x), self.selection_model(x)], axis=1)


class HeckmanDGBinaryLinearClassifier:
    def __init__(self, config=dict()):

        self.config = {
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_epoch': 100,
            'batch_size': 1000
        }

        self.config.update(config)

    def fit(self, data):
        dataloaders = dict()
        for split in ['train', 'valid']:
            dataloaders[split] = DataLoader(
                TensorDataset(
                    torch.FloatTensor(data[f'{split}_x']),
                    torch.FloatTensor(data[f'{split}_y']),
                    torch.FloatTensor(data[f'{split}_s'])
                ),
                shuffle=(split == 'train'), batch_size=self.config['batch_size'], drop_last=True
            )

        n_covariates = data['train_x'].shape[1]
        n_domains = data['train_s'].shape[1]
        self.model = LinearHeckmanDG(n_covariates, n_domains).to(self.config['device'])

        opt = torch.optim.AdamW(
            [{'params': self.model.outcome_model.parameters()},
             {'params': self.model.selection_model.parameters()},
             {'params': self.model.rho, 'lr': 2e-2, 'weight_decay': 0.}],
            lr=1e-3, weight_decay=0.
        )

        sch = None

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        rho_traj = []
        best_model, best_loss, best_acc = deepcopy(self.model.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.model.train()
            train_loss = 0.
            train_pred, train_true = [], []
            for batch in dataloaders['train']:
                batch = [b.to(self.config['device']) for b in batch]

                batch_probits = self.model(batch[0])
                batch_prob = normal.cdf(batch_probits[:, 0])

                loss = 0.
                for j in range(n_domains):
                    joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j + 1], self.model.rho[j])
                    loss += -(batch[1] * batch[2][:, j] * safe_log(joint_prob)).mean()
                    loss += -((1 - batch[1]) * batch[2][:, j] * safe_log(
                        normal.cdf(batch_probits[:, j + 1]) - joint_prob)).mean()
                    loss += -((1 - batch[2][:, j]) * safe_log(normal.cdf(-batch_probits[:, j + 1]))).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.model.rho.data = torch.clip(self.model.rho.data, -0.99, 0.99)

                train_loss += loss.item() / len(dataloaders['train'])
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(batch[1].detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.concatenate(train_true),
                    np.concatenate(train_pred)))

            rho_traj.append(self.model.rho.data.detach().cpu().numpy())

            if sch:
                sch.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                for batch in dataloaders['valid']:
                    batch = [b.to(self.config['device']) for b in batch]
                    batch_probits = self.model(batch[0])
                    batch_prob = normal.cdf(batch_probits[:, 0])

                    loss = 0.
                    for j in range(n_domains):
                        joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], self.model.rho[j])
                        loss += -(batch[1] * batch[2][:, j] * safe_log(joint_prob)).mean()
                        loss += -((1 - batch[1]) * batch[2][:, j] * safe_log(normal.cdf(batch_probits[:, j+1]) - joint_prob)).mean()
                        loss += -((1 - batch[2][:, j]) * safe_log(1. - normal.cdf(batch_probits[:, j+1]))).mean()

                    valid_loss += loss.item() / len(dataloaders['valid'])
                    valid_pred.append(batch_prob.detach().cpu().numpy())
                    valid_true.append(batch[1].detach().cpu().numpy())

            if valid_loss < best_loss:
                best_model = deepcopy(self.model.state_dict())
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

        self.model.load_state_dict(best_model)
        self.train_loss_traj = train_loss_traj
        self.valid_loss_traj = valid_loss_traj
        self.train_auc_traj = train_auc_traj
        self.valid_auc_traj = valid_auc_traj
        self.rho_traj = rho_traj
        self.best_train_loss = train_loss_traj[best_epoch]
        self.best_valid_loss = valid_loss_traj[best_epoch]
        self.best_train_auc = train_auc_traj[best_epoch]
        self.best_valid_auc = valid_auc_traj[best_epoch]
        self.best_rho = rho_traj[best_epoch]
        self.best_epoch = best_epoch

    def predict_proba(self, X):
        dataloader = DataLoader(
            TensorDataset(torch.FloatTensor(X)),
            shuffle=False, batch_size=self.config['batch_size'], drop_last=False
        )

        normal = torch.distributions.normal.Normal(0, 1)
        with torch.no_grad():
            probs = torch.cat(
                [normal.cdf(self.model(batch_x.to(self.config['device']))[:, 0]) for (batch_x, ) in dataloader]
            ).detach().cpu().numpy()

        return probs

    def get_selection_probit(self, data):
        self.model.eval()
        dataloader = DataLoader(
            TensorDataset(
                torch.FloatTensor(data['train_x']),
                torch.FloatTensor(data['train_s'])
            ),
            shuffle=False, batch_size=self.config['batch_size'], drop_last=False
        )

        with torch.no_grad():
            probits = torch.cat(
                [self.model(batch[0].to(self.config['device']))[:, 1:] for batch in dataloader]
            ).detach().cpu().numpy()
            labels = data['train_s'].argmax(1)

        return probits, labels

