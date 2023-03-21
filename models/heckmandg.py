
import torch
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
        (1 / (2 * np.pi)) * torch.trapezoid(y=y, x=x)


class HeckmanDGBinaryClassifier:
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
                    torch.FloatTensor(data[f'{split}_y']),
                    torch.FloatTensor(data[f'{split}_s'])
                ),
                shuffle=(split == 'train'), batch_size=self.config['batch_size'], drop_last=True
            )

        n_domains = data['train_s'].shape[1]
        try:
            opt = self.optimizer(
                [{'params': self.network.layers.parameters()},
                 {'params': self.network.rho, 'lr': 1e-2, 'weight_decay': 0.}]
            )
        except:
            opt = self.optimizer(
                [{'params': self.network.f_layers.parameters()},
                 {'params': self.network.g_layers.parameters()},
                 {'params': self.network.rho, 'lr': 1e-2, 'weight_decay': 0.}]
            )
        sch = self.scheduler(opt) if self.scheduler else None

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        rho_traj = []
        best_model, best_loss, best_acc = deepcopy(self.network.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.network.train()
            train_loss = 0.
            train_pred, train_true = [], []
            for batch in dataloaders['train']:
                batch = [b.to(self.config['device']) for b in batch]

                batch_probits = self.network(batch[0])
                batch_prob = normal.cdf(batch_probits[:, 0])

                loss = 0.
                for j in range(n_domains):
                    joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j + 1], self.network.rho[j])
                    loss += -(batch[1] * batch[2][:, j] * safe_log(joint_prob)).mean()
                    loss += -((1 - batch[1]) * batch[2][:, j] * safe_log(
                        normal.cdf(batch_probits[:, j + 1]) - joint_prob)).mean()
                    loss += -((1 - batch[2][:, j]) * safe_log(normal.cdf(-batch_probits[:, j + 1]))).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)

                train_loss += loss.item() / len(dataloaders['train'])
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(batch[1].detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.concatenate(train_true),
                    np.concatenate(train_pred)))

            rho_traj.append(self.network.rho.data.detach().cpu().numpy())

            if sch:
                sch.step()

            self.network.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                for batch in dataloaders['valid']:
                    batch = [b.to(self.config['device']) for b in batch]
                    batch_probits = self.network(batch[0])
                    batch_prob = normal.cdf(batch_probits[:, 0])

                    loss = 0.
                    for j in range(n_domains):
                        joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], self.network.rho[j])
                        loss += -(batch[1] * batch[2][:, j] * safe_log(joint_prob)).mean()
                        loss += -((1 - batch[1]) * batch[2][:, j] * safe_log(normal.cdf(batch_probits[:, j+1]) - joint_prob)).mean()
                        loss += -((1 - batch[2][:, j]) * safe_log(1. - normal.cdf(batch_probits[:, j+1]))).mean()

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
                [normal.cdf(self.network(batch_x.to(self.config['device']))[:, 0]) for (batch_x, ) in dataloader]
            ).detach().cpu().numpy()

        return probs

    def get_selection_probit(self, data):
        self.network.eval()
        dataloader = DataLoader(
            TensorDataset(
                torch.FloatTensor(data['train_x']),
                torch.FloatTensor(data['train_s'])
            ),
            shuffle=False, batch_size=self.config['batch_size'], drop_last=False
        )

        with torch.no_grad():
            probits = torch.cat(
                [self.network(batch[0].to(self.config['device']))[:, 1:] for batch in dataloader]
            ).detach().cpu().numpy()
            labels = data['train_s'].argmax(1)

        return probits, labels

class HeckmanDGBinaryClassifierCNN:
    def __init__(self, args,
                 network, optimizer, scheduler, 
                 ):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.network = self.network.to(self.args.device)

    def fit(self, 
            train_loader, 
            valid_loader, 
            ):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        opt = self.optimizer(
            [{'params': self.network.f_layers.parameters()},
                {'params': self.network.g_layers.parameters()},
                {'params': self.network.rho, 'lr': 1e-2, 'weight_decay': 0.}]
        )
        sch = self.scheduler(opt) if self.scheduler else None

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        rho_traj = []
        best_model, best_loss, best_acc = deepcopy(self.network.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            self.network.train()
            train_loss = 0.
            train_pred, train_true = [], []
            ##### train_dataloader
            for b, batch in enumerate(self.train_loader):
                print('train batch: ', b, f'/ {len(train_loader)}' )
                
                x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)

 
                batch_probits = self.network(x) # shape(batch, output of f_layers+g_layers) , 
                batch_prob = normal.cdf(batch_probits[:, 0]) # 0: f_layers

                loss = 0.
                for j in range(self.args.num_domains):
                    joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j + 1], self.network.rho[j])
                    loss += -(y * s[:, j] * safe_log(joint_prob)).mean()
                    loss += -((1 - y) * s[:, j] * safe_log(
                        normal.cdf(batch_probits[:, j + 1]) - joint_prob)).mean()
                    loss += -((1 - s[:, j]) * safe_log(normal.cdf(-batch_probits[:, j + 1]))).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)

                train_loss += loss.item() / len(self.train_loader)
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(y.detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.concatenate(train_true),
                    np.concatenate(train_pred)))

            rho_traj.append(self.network.rho.data.detach().cpu().numpy())

            if sch:
                sch.step()

            self.network.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                ##### valid_dataloader
                for b, batch in enumerate(self.valid_loader):
                    print('valid batch: ', b, f'/ {len(valid_loader)}' )
                
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)

                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)

                    batch_probits = self.network(x)
                    batch_prob = normal.cdf(batch_probits[:, 0])

                    loss = 0.
                    for j in range(self.args.num_domains):
                        joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], self.network.rho[j])
                        loss += -(y * s[:, j] * safe_log(joint_prob)).mean()
                        loss += -((1 - y) * s[:, j] * safe_log(normal.cdf(batch_probits[:, j+1]) - joint_prob)).mean()
                        loss += -((1 - s[:, j]) * safe_log(1. - normal.cdf(batch_probits[:, j+1]))).mean()

                    valid_loss += loss.item() / len(valid_loader)
                    valid_pred.append(batch_prob.detach().cpu().numpy())
                    valid_true.append(y.detach().cpu().numpy())

            if valid_loss < best_loss:
                best_model = deepcopy(self.network.state_dict())
                best_loss = valid_loss
                best_epoch = epoch

            valid_loss_traj.append(valid_loss)
            valid_auc_traj.append(
                roc_auc_score(
                    np.concatenate(valid_true),
                    np.concatenate(valid_pred)))

            desc = f'[{epoch + 1:03d}/{self.args.epochs}] '
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
        self.rho_traj = rho_traj
        self.best_train_loss = train_loss_traj[best_epoch]
        self.best_valid_loss = valid_loss_traj[best_epoch]
        self.best_train_auc = train_auc_traj[best_epoch]
        self.best_valid_auc = valid_auc_traj[best_epoch]
        self.best_rho = rho_traj[best_epoch]
        self.best_epoch = best_epoch

    def predict_proba(self, batch):
        normal = torch.distributions.normal.Normal(0, 1)
        x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
        y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
        with torch.no_grad():
            probs_batch = normal.cdf(self.network(x.to(self.args.device))[:, 0])
        probs_batch = probs_batch.detach().cpu().numpy()
        return probs_batch
    
    def predict_proba_loader(self, dataloader):
        normal = torch.distributions.normal.Normal(0, 1)
        probs_list=[]
        for b, batch in enumerate(dataloader):
            print('predict batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
            
            with torch.no_grad():
                probs_batch = normal.cdf(self.network(x.to(self.args.device))[:, 0])
                probs_list.append(probs_batch)
                
        probs = torch.cat(probs_list).detach().cpu().numpy()

        return probs

    def get_selection_probit(self, dataloader):
        self.network.eval()
        probits_list = []
        for b, batch in enumerate(dataloader):
            print('get_selection_probit batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
            s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
            one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
            col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
            for i in range(len(one_hot)):
                one_hot[i,col_idx[i]] = 1
            s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
            # s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)

            with torch.no_grad():
                probits_batch = self.network(x.to(self.args.device))[:, 1:] 
                probits_list.append(probits_batch)
                        
        probits = torch.cat(probits_list).detach().cpu().numpy()
        labels = s.argmax(1)
        labels = labels.detach().cpu().numpy()
        return probits, labels

class TwoStepHeckmanDGBinaryClassifier:
    def __init__(self, network, optimizer, scheduler, config=dict()):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = {
            'device': 'cuda',
            'max_epoch': 100,
            'batch_size': 1000,
        }

        self.config.update(config)
        self.network = self.network.to(self.config['device'])

    def fit_selection_model(self, data):
        dataloaders = dict()
        for split in ['train', 'valid']:
            dataloaders[split] = DataLoader(
                TensorDataset(
                    torch.FloatTensor(data[f'{split}_x']),
                    torch.FloatTensor(data[f'{split}_s'])
                ),
                shuffle=(split == 'train'), batch_size=self.config['batch_size'], drop_last=False
            )

        opt = self.optimizer(params=self.network.g_layers.parameters())
        sch = self.scheduler(opt) if self.scheduler else None

        for params in self.network.f_layers.parameters():
            params.requires_grad = False

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        best_model, best_loss, best_acc = deepcopy(self.network.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.network.g_layers.train()
            self.network.f_layers.eval()
            train_loss = 0.
            train_pred, train_true = [], []
            for batch in dataloaders['train']:
                batch = [b.to(self.config['device']) for b in batch]

                batch_probits = self.network.forward_g(batch[0])
                batch_prob = normal.cdf(batch_probits)

                loss = -(batch[1]*safe_log(batch_prob) + (1-batch[1])*safe_log(1 - batch_prob)).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() / len(dataloaders['train'])
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(batch[1].detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.row_stack(train_true),
                    np.row_stack(train_pred),
                    average='macro')
            )

            if sch:
                sch.step()

            self.network.g_layers.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                for batch in dataloaders['valid']:
                    batch = [b.to(self.config['device']) for b in batch]

                    batch_probits = self.network.forward_g(batch[0])
                    batch_prob = normal.cdf(batch_probits)

                    loss = -(batch[1] * safe_log(batch_prob) + (1 - batch[1]) * safe_log(1 - batch_prob)).mean()

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
                    np.concatenate(valid_pred),
                    average='macro')
            )

            desc = f'[{epoch + 1:03d}/{self.config["max_epoch"]:03d}] '
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            desc += f'AUROC: {train_auc_traj[-1]:.4f} '
            desc += f'| valid | Loss: {valid_loss_traj[-1]:.4f} '
            desc += f'AUROC: {valid_auc_traj[-1]:.4f} '
            pbar.set_description(desc)

        self.network.load_state_dict(best_model)
        print(f'selection model best at {best_epoch} with loss {valid_loss_traj[best_epoch]:.4f} auc {valid_auc_traj[best_epoch]:.4f}')

    def fit(self, data):
        self.fit_selection_model(data)

        dataloaders = dict()
        for split in ['train', 'valid']:
            
            dataloaders[split] = DataLoader(
                TensorDataset(
                    torch.FloatTensor(data[f'{split}_x']), # batch[0]
                    torch.FloatTensor(data[f'{split}_y']), # batch[1]
                    torch.FloatTensor(data[f'{split}_s']) # batch[2]
                ),
                shuffle=(split == 'train'), batch_size=self.config['batch_size'], drop_last=False
            )

        n_domains = data['train_s'].shape[1]
        opt = self.optimizer(
            [{'params': self.network.f_layers.parameters()},
             {'params': self.network.rho, 'lr': 1e-2, 'weight_decay': 0.}]
        )
        sch = self.scheduler(opt) if self.scheduler else None

        for params in self.network.f_layers.parameters():
            params.requires_grad = True

        for params in self.network.g_layers.parameters():
            params.requires_grad = False

        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        rho_traj = []
        best_model, best_loss, best_acc = deepcopy(self.network.state_dict()), 1e10, 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.config['max_epoch']))
        for epoch in pbar:
            self.network.f_layers.train()
            self.network.g_layers.eval()
            train_loss = 0.
            train_pred, train_true = [], []
            for batch in dataloaders['train']:
                batch = [b.to(self.config['device']) for b in batch]

                batch_probits = self.network(batch[0])
                batch_prob = normal.cdf(batch_probits[:, 0])

                loss = 0.
                for j in range(n_domains):
                    joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], self.network.rho[j])
                    loss += -(batch[1]*batch[2][:, j]*safe_log(joint_prob)).mean()
                    loss += -((1-batch[1])*batch[2][:, j]*safe_log(normal.cdf(batch_probits[:, j+1]) - joint_prob)).mean()
                    loss += -((1-batch[2][:, j])*safe_log(normal.cdf(-batch_probits[:, j+1]))).mean()

                opt.zero_grad()
                loss.backward()

                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)

                train_loss += loss.item() / len(dataloaders['train'])
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(batch[1].detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            train_auc_traj.append(
                roc_auc_score(
                    np.concatenate(train_true),
                    np.concatenate(train_pred)))

            rho_traj.append(self.network.rho.data.detach().cpu().numpy())

            if sch:
                sch.step()

            self.network.f_layers.eval()
            self.network.g_layers.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_pred, valid_true = [], []
                for batch in dataloaders['valid']:
                    batch = [b.to(self.config['device']) for b in batch]
                    batch_probits = self.network(batch[0])
                    batch_prob = normal.cdf(batch_probits[:, 0])

                    loss = 0.
                    for j in range(n_domains):
                        joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits[:, j+1], self.network.rho[j])
                        loss += -(batch[1] * batch[2][:, j] * safe_log(joint_prob)).mean()
                        loss += -((1 - batch[1]) * batch[2][:, j] * safe_log(normal.cdf(batch_probits[:, j + 1]) - joint_prob)).mean()
                        loss += -((1 - batch[2][:, j]) * safe_log(normal.cdf(-batch_probits[:, j + 1]))).mean()

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
                [normal.cdf(self.network(batch_x.to(self.config['device']))[:, 0]) for (batch_x, ) in dataloader]
            ).detach().cpu().numpy()

        return probs

    def get_selection_probit(self, data):
        self.network.eval()
        dataloader = DataLoader(
            TensorDataset(
                torch.FloatTensor(data['train_x']),
                torch.FloatTensor(data['train_s'])
            ),
            shuffle=False, batch_size=self.config['batch_size'], drop_last=False
        )

        with torch.no_grad():
            probits = torch.cat(
                [self.network(batch[0].to(self.config['device']))[:, 1:] for batch in dataloader]
            ).detach().cpu().numpy()
            labels = data['train_s'].argmax(1)

        return probits, labels