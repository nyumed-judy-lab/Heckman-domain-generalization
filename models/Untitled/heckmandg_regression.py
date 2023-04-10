import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils_datasets.transforms import InputTransforms

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))

class HeckmanDG_CNN_Regressor:
    def __init__(self, 
                 args,
                 network, 
                 optimizer, 
                 scheduler = None, 
                 ):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.network = self.network.to(self.args.device)
        # self.network = network.to(args.device)
        
        InputTransformObj: object = InputTransforms[self.args.data]
        self.train_transform = InputTransformObj(augmentation= self.args.augmentation, randaugment= self.args.randaugment,)
        self.eval_transform = InputTransformObj(augmentation=False)
        self.train_transform.to(self.args.device)
        self.eval_transform.to(self.args.device)
        
        '''
        InputTransformObj: object = InputTransforms[args.data]
        train_transform = InputTransformObj(augmentation= args.augmentation, randaugment=args.randaugment,)
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
        sch = self.scheduler(opt) if self.scheduler else None
        '''
        scheduler = none
        opt = optimizer
        sch = scheduler(opt) if scheduler else None
        '''

        train_loss_traj, valid_loss_traj = [], []
        # regression evaluation
        train_pearsonr_traj, valid_pearsonr_traj = [], []
        train_mse_traj, valid_mse_traj = [], []
        train_mae_traj, valid_mae_traj = [], []
        train_mape_traj, valid_mape_traj = [], []
        rho_traj = []
        # regression only
        sigma_traj = []
        
        ##################################################
        best_model, best_loss, best_pearsonr = deepcopy(self.network.state_dict()), 1e10, 0.
        '''
        best_model, best_loss, best_pearsonr = deepcopy(network.state_dict()), 1e10, 0.
        '''
        ##################################################
        epsilon: float = 1e-5
        normal = torch.distributions.normal.Normal(0, 1)

        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            # epoch = 0
            self.network.train()
            # network.train()
            train_loss = 0.
            train_pred, train_true = [], []
            ##### train_dataloader
            for b, batch in enumerate(self.train_loader):
                print('train batch: ', b, f'/ {len(train_loader)}' )
                '''
                for b, batch in enumerate(train_loader):
                    print('train batch: ', b, f'/ {len(train_loader)}' )
                '''
                
                x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))
                y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                '''
                x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                if train_transform is not None:
                    x = train_transform(x.to(torch.uint8))
                y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
                one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                '''
                
                x = x.float().to(torch.float32)
                self.network = self.network.to(self.args.device)
                
                batch_probits = self.network(x)
                batch_probits_f = self.network.forward_f(x).squeeze().to(self.args.device)
                batch_probits_g = self.network.forward_g(x).to(self.args.device)
                '''
                network = network.to(args.device)
                batch_probits = network.forward(x)
                batch_probits_f = network.forward_f(x).squeeze().to(args.device)
                batch_probits_g = network.forward_g(x).to(args.device)
                '''

                ##################################################
                loss = 0.
                for k in range(self.args.num_domains):
                    # k=0
                    y_true = y
                    s_true = s[:,k]
                    y_pred = batch_probits_f
                    s_pred = batch_probits_g[:,k]
                    
                    rho = self.network.rho[k]
                    sigma = self.network.sigma

                    epsilon: float = 1e-5
                    normal = torch.distributions.Normal(loc=0., scale=1.)
                    # A) Selection loss for unselected (unlabeled) individuals: (N,  ); Loss takes value of zero for S=1
                    loss_not_selected = - (1 - s_true) * torch.log(1 - normal.cdf(s_pred) + epsilon)
                    # B) Selection loss for selected (labeled) individuals: (N,  )
                    probit_loss = - torch.log(
                        normal.cdf(
                            (s_pred + rho * (y_true - y_pred).div(sigma)).div(torch.sqrt(1 - rho ** 2))
                        ) + epsilon
                    )
                    # C) Regression loss for selected individuals: (N,  )
                    regression_loss = 0.5 * (
                        torch.log(2 * torch.pi * (sigma ** 2)) \
                            + F.mse_loss(y_pred, y_true, reduction='none').div(sigma ** 2)
                    )
                    loss += (loss_not_selected + s_true * (probit_loss + regression_loss)).mean()

                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)

                train_loss += loss.item() / len(self.train_loader) # mean
                train_pred.append(y_pred.detach().cpu().numpy())
                train_true.append(y_true.detach().cpu().numpy())

            train_loss_traj.append(train_loss)
            pers, pval = pearsonr(np.concatenate(train_true),
                                  np.concatenate(train_pred)
                                  )
            train_pearsonr_traj.append(pers)

            rho_traj.append(self.network.rho.data.detach().cpu().numpy())
            sigma_traj.append(self.network.sigma.data.detach().cpu().numpy())

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

                    x = x.float().to(torch.float32)
                    self.network = self.network.to(self.args.device)
                    
                    batch_probits = self.network(x)
                    batch_probits_f = self.network.forward_f(x).squeeze().to(self.args.device)
                    batch_probits_g = self.network.forward_g(x).to(self.args.device)
                    '''
                    network = network.to(args.device)
                    batch_probits = network.forward(x)
                    batch_probits_f = network.forward_f(x).squeeze().to(args.device)
                    batch_probits_g = network.forward_g(x).to(args.device)
                    '''
                    loss = 0.
                    for k in range(self.args.num_domains):
                        # k=0
                        y_true = y
                        s_true = s[:,k]
                        y_pred = batch_probits_f
                        s_pred = batch_probits_g[:,k]
                        rho = self.network.rho[k]
                        sigma = self.network.sigma

                        epsilon: float = 1e-5
                        normal = torch.distributions.Normal(loc=0., scale=1.)
                        # A) Selection loss for unselected (unlabeled) individuals: (N,  ); Loss takes value of zero for S=1
                        loss_not_selected = - (1 - s_true) * torch.log(1 - normal.cdf(s_pred) + epsilon)
                        # B) Selection loss for selected (labeled) individuals: (N,  )
                        probit_loss = - torch.log(
                            normal.cdf(
                                (s_pred + rho * (y_true - y_pred).div(sigma)).div(torch.sqrt(1 - rho ** 2))
                            ) + epsilon
                        )
                        # C) Regression loss for selected individuals: (N,  )
                        regression_loss = 0.5 * (
                            torch.log(2 * torch.pi * (sigma ** 2)) \
                                + F.mse_loss(y_pred, y_true, reduction='none').div(sigma ** 2)
                        )
                        loss += (loss_not_selected + s_true * (probit_loss + regression_loss)).mean()
                    valid_loss += loss.item() / len(valid_loader)
                    valid_pred.append(y_pred.detach().cpu().numpy())
                    valid_true.append(y_true.detach().cpu().numpy())

            if valid_loss < best_loss:
                best_model = deepcopy(self.network.state_dict())
                best_loss = valid_loss
                best_epoch = epoch
            valid_loss_traj.append(valid_loss)
            pers, pval = pearsonr(np.concatenate(valid_true),
                                  np.concatenate(valid_pred)
                                  )
            valid_pearsonr_traj.append(pers)

            desc = f'[{epoch + 1:03d}/{self.args.epochs}] '
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            desc += f'pearsonr: {train_pearsonr_traj[-1]:.4f} '
            desc += f'| valid | Loss: {valid_loss_traj[-1]:.4f} '
            desc += f'pearsonr: {valid_pearsonr_traj[-1]:.4f} '
            pbar.set_description(desc)

        self.network.load_state_dict(best_model)
        
        self.train_loss_traj = train_loss_traj
        self.valid_loss_traj = valid_loss_traj
        self.train_pearsonr_traj = train_pearsonr_traj
        self.valid_pearsonr_traj = valid_pearsonr_traj
        self.rho_traj = rho_traj

        self.best_train_loss = train_loss_traj[best_epoch]
        self.best_valid_loss = valid_loss_traj[best_epoch]
        self.best_train_pearsonr = train_pearsonr_traj[best_epoch]
        self.best_valid_pearsonr = valid_pearsonr_traj[best_epoch]
        self.best_rho = rho_traj[best_epoch]
        self.best_sigma = sigma_traj[best_epoch]
        self.best_epoch = best_epoch

    def predict(self, batch):
        # normal = torch.distributions.normal.Normal(0, 1)
        x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
        with torch.no_grad():
            y_pred_batch = self.network.forward_f(x.to(self.args.device))[:, 0]
            # network.forward_f(x.cpu())[:,0]
        y_pred_batch = y_pred_batch.detach().cpu().numpy()
        return y_pred_batch
    
    def predict_loader(self, dataloader):
        y_pred_list=[]
        for b, batch in enumerate(dataloader):
            print('predict batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            
            with torch.no_grad():
                y_pred_batch = self.network.forward_f(x.to(self.args.device))[:, 0]
                y_pred_list.append(y_pred_batch)
        y_preds = torch.cat(y_pred_list).detach().cpu().numpy()

        return y_preds

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
