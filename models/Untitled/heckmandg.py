import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils_datasets.transforms import InputTransforms
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
                '''
                Equation 20 in ICLR paper
                '''
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
        
        InputTransformObj: object = InputTransforms[self.args.data]
        self.train_transform = InputTransformObj(augmentation= self.args.augmentation, randaugment=self.args.randaugment,)
        self.eval_transform = InputTransformObj(augmentation=False)
        self.train_transform.to(self.args.device)
        self.eval_transform.to(self.args.device)



    def fit(self, 
            train_loader, 
            valid_loader, 
            ):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        opt = self.optimizer(
            [{'params': self.network.f_layers.parameters()},
                {'params': self.network.g_layers.parameters()},
                {'params': self.network.rho, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay}]
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
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))

                
                y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)


 
                # batch_probits = self.network(x) # shape(batch, output of f_layers+g_layers) , 
                batch_probits = self.network.forward(x) # (batch, #output_f_layers + #output_g_layers) 
                batch_probits_f = self.network.forward_f(x).squeeze()
                batch_probits_g = self.network.forward_g(x)
                # batch_prob = normal.cdf(batch_probits[:, 0]) # 0: f_layers
                batch_prob = normal.cdf(batch_probits_f) # 0: f_layers  

                '''
                Equation 20 in ICLR paper
                '''
                loss = 0.
                for k in range(self.args.num_domains):
                    # k=0
                    # joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits_g[:, k], self.network.rho[k])
                    joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
                    loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
                    loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                    loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()

                if False:
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
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))

                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)

                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)

                    batch_probits = self.network(x)
                    batch_prob = normal.cdf(batch_probits[:, 0])
                    '''
                    Equation 20 in ICLR paper
                    '''
                    loss = 0.
                    for k in range(self.args.num_domains):
                        # k=0
                        # joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits_g[:, k], self.network.rho[k])
                        joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
                        loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
                        loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                        loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()
                    if False:
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
    
    def fit(self, 
            train_loader, 
            valid_loader, 
            ):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        opt = self.optimizer(
            [{'params': self.network.f_layers.parameters()},
                {'params': self.network.g_layers.parameters()},
                {'params': self.network.rho, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay}]
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
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))

                
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
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))

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

class HeckmanDG_CNN_Classifier:
    def __init__(self, 
                 args,
                 network, 
                 optimizer, 
                 scheduler=None, 
                 ):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.network = network.to(args.device)
        
        self.InputTransformObj: object = InputTransforms[self.args.data]
        self.train_transform = self.InputTransformObj(augmentation=self.args.augmentation, randaugment=self.args.randaugment)
        self.eval_transform = self.InputTransformObj(augmentation=False)
        self.train_transform.to(self.args.device)
        self.eval_transform.to(self.args.device)
        
        '''
        InputTransformObj: object = InputTransforms[args.data]
        train_transform = InputTransformObj(augmentation=args.augmentation, randaugment=args.randaugment)
        eval_transform = InputTransformObj(augmentation=False)
        train_transform.to(args.device)
        eval_transform.to(args.device)
        '''

    def fit(self, 
            train_loader, 
            valid_loader, 
            ):
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        opt = self.optimizer
        '''
        opt = optimizer
        '''
        # opt = optimizer(
        #     [{'params': network.f_layers.parameters()},
        #      {'params': network.g_layers.parameters()},
        #      {'params': network.rho, 'lr': args.lr, 'weight_decay': args.weight_decay}]
        # )
        # scheduler = None
        # sch = scheduler(opt) if scheduler else None
        
        train_loss_traj, valid_loss_traj = [], []
        train_auc_traj, valid_auc_traj = [], []
        ########################################
        train_acc_traj, valid_f1_traj = [], []
        train_f1_traj, valid_f1_traj = [], []
        ########################################
        rho_traj = []
        best_model, best_loss, best_auc, best_acc, best_f1 = deepcopy(network.state_dict()), 1e10, 0., 0., 0.
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(args.epochs))
        for epoch in pbar:
            # epoch = 0
            self.network.train() #self.
            train_loss = 0.
            train_pred, train_true = [], []

            ##### train_dataloader
            for b, batch in enumerate(train_loader):
                # self.
                print('train batch: ', b, f'/ {len(train_loader)}' )
                
                x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))
                
                
                y_ = batch['y'].to(self.args.device).to(torch.float32)  # (Batch_size)
                if self.args.loss_type == 'multiclass':
                    one_hot = np.zeros((self.args.batch_size, len(self.args.num_classes)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.num_classes))[0][0] for idx in range(len(y_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    y = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (Batch_size, # Nomains)
                else:
                    y = y_
                    
                ##########
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (Batch_size)
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (Batch_size, # Nomains)
 
                batch_probits = self.network(x) # batch_probits.shape 
                batch_probits = self.network.forward(x) # (batch, #output_f_layers + #output_g_layers) 
                batch_probits_f = self.network.forward_f(x)
                batch_probits_g = self.network.forward_g(x)
                
                if args.loss_type == 'multiclass':
                    batch_prob = batch_probits_f.clone()
                    for c in range(batch_probits_f.shape[1]):
                        batch_prob[:,c] = normal.cdf(batch_prob[:,c]) # f_layers
                else:
                    batch_prob = normal.cdf(batch_probits_f) # 0: f_layers
                
                if args.loss_type == 'multiclass':
                    # binary classification
                    loss = 0.
                    for j in range(args.num_classes):
                        for k in range(args.num_domains):
                            # j = 0
                            batch_probits_f.shape
                            batch_probits_g.shape
                            network.rho.shape
                            joint_prob = bivariate_normal_cdf(batch_probits_f[:,j],
                                                            batch_probits_g[:,k],
                                                            network.rho[k])
                            loss += -(y * s[:, k] * safe_log(joint_prob)).mean()
                            loss += -((1 - y) * s[:, k] * safe_log(
                                normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                            loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()
                else:
                    # binary classification
                    loss = 0.
                    for k in range(args.num_domains):
                        #k=0
                        joint_prob = bivariate_normal_cdf(batch_probits[:, 0], 
                                                        batch_probits[:, k + 1], 
                                                        network.rho[k])
                        network.rho.shape
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
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))

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

def multivariate_normal_cdf(a: torch.Tensor, b: torch.Tensor, rho: torch.FloatTensor, steps: int = 100):
    """
    Approximation of standard multivariate normal cdf using the trapezoid rule.
    Arguments:
        a: 1D tensor of shape (B, )
        b: 1D tensor of shape (B, )
        rho: 2D tensor of shape (B, C), where C is the number of classes
    """
    device = a.device
    a, b = a.view(-1, 1), b.view(-1, 1)  # for proper broadcasting with x
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    C = rho.shape[1]
    x = _linspace_with_grads(start=0, stop=rho.max(), steps=steps,
                             device=device).unsqueeze(0).expand(a.shape[0], -1)  # (B, steps)
    y = torch.zeros((a.shape[0], C), device=device)
    for i in range(C):
        z = (1 - rho[:, i]**2)**(-0.5) * (rho[:, i]*x - a)
        y[:, i] = torch.exp(-0.5*(torch.pow(z, 2) + torch.pow(b, 2))/(1-torch.pow(x, 2))).sum(dim=1)
    y *= (1 / (2 * np.pi)) * (1 / steps) * rho.prod(dim=1).view(-1, 1)
    
    return _normal.cdf(a.squeeze()) * _normal.cdf(b.squeeze()) + y.sum(dim=1)

import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from utils_datasets.transforms import InputTransforms
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

if False:
    
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import typing
    from torch.utils.data import ConcatDataset, DataLoader
    from networks.cnn_initializers import NetworkInitializer


    class HeckmanCNN(nn.Module):
        def __init__(self, args):
            self.args = args

            super(HeckmanCNN, self).__init__()
            g_layers_backbone = NetworkInitializer.initialize_backbone(
                name=args.backbone, data=args.data, pretrained=args.pretrained,
            )
            g_layers_head = nn.Linear(
                in_features=g_layers_backbone.out_features, out_features=args.num_domains,
            )
            g_layers_ = list(self.g_layers_backbone.children())
            g_layers_.append(self.g_layers_head)
            self.g_layers = nn.Sequential(*g_layers_)

            # Outcome model: backbone
            self.f_layers_backbone = NetworkInitializer.initialize_backbone(
                    name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
            )
            # Outcome model: head
            self.f_layers_head = nn.Linear(
                in_features=self.f_layers_backbone.out_features,
                out_features=NetworkInitializer._out_features[self.args.data][0],
            )
            f_layers_ = list(self.f_layers_backbone.children())
            f_layers_.append(self.f_layers_head)
            self.f_layers = nn.Sequential(*f_layers_)
            
            self.rho = nn.Parameter(torch.zeros(args.num_domains, dtype=torch.float), requires_grad=True)
            self.register_parameter(name='rho', param=self.rho)

        def forward_f(self, x):
            return self.f_layers(x)

        def forward_g(self, x):
            return self.g_layers(x)

        def forward(self, x):
            return torch.cat([self.f_layers(x), self.g_layers(x)], axis=1)

        # def forward_f(self, x):
        #     x = self.f_layers_backbone(x)
        #     x = torch.cat([x, self.f_layers_head(x)], dim=1)
        #     return x #self.f_layers(x)

        # def forward_g(self, x):
        #     x = self.g_layers_backbone(x)
        #     x = torch.cat([x, self.g_layers_head(x)], dim=1)
        #     return x #self.g_layers(x)
    network = HeckmanCNN(args)
