import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils_datasets.transforms import InputTransforms

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))

def loss_regression(self,
                    y,
                    s,
                    batch_probits_f,
                    batch_probits_g,
                    ):
    # safe_log = lambda x: torch.log(torch.clip(x, 1e-6))
    # normal = torch.distributions.normal.Normal(0, 1)
    epsilon: float = 1e-5
    normal = torch.distributions.Normal(loc=0., scale=1.)
    loss = 0.
    y_true = y.clone()
    y_pred = batch_probits_f.clone()
    
    for k in range(self.args.num_domains):
        # k=0
        s_true = s[:,k].clone()
        s_pred = batch_probits_g[:,k].clone()
        rho = self.network.rho[k]
        sigma = self.network.sigma
        # s_true = s[:,k]
        # rho = network.rho[k] # rho.shape
        # sigma = network.sigma
        # sigma.shape

        # Equation 19 - 2
        # A) Selection loss for unselected (unlabeled) individuals: (N,  ); Loss takes value of zero for S=1
        loss_not_selected = (1 - s_true) * torch.log(normal.cdf(-s_pred) + epsilon)
        if False:
            loss_not_selected = - (1 - s_true) * torch.log(1 - normal.cdf(s_pred) + epsilon)
        # Equation 19 - 1 - 1
        # B) Selection loss for selected (labeled) individuals: (N,  )
        probit_loss = torch.log(
            normal.cdf(
                (s_pred + rho * (y_true - y_pred).div(sigma)).div(torch.sqrt(1 - rho ** 2))
            ) + epsilon
        )
        if False:
            probit_loss = - torch.log(
                normal.cdf(
                    (s_pred + rho * (y_true - y_pred).div(sigma)).div(torch.sqrt(1 - rho ** 2))
                ) + epsilon
            )
        # Equation 19 - 1 - 2
        # C) Regression loss for selected individuals: (N,  )
        regression_loss = -0.5*(torch.log(2 * torch.pi * (sigma ** 2))) -0.5*(((y_pred - y_true).div(sigma))**2)
        if False:
            regression_loss = 0.5 * (
                torch.log(2 * torch.pi * (sigma ** 2)) \
                    + F.mse_loss(y_pred, y_true, reduction='none').div(sigma ** 2)
            )
            # _epsilon + sigma
        # Equation 19 
        loss += ((-1 * loss_not_selected) + (-1 * s_true * (probit_loss + regression_loss))).mean()
        if False:
            loss += (loss_not_selected + s_true * (probit_loss + regression_loss)).mean()
        return loss
    
class HeckmanDG_CNN_Regressor:
    def __init__(self, 
                 args,
                 network, 
                 optimizer, 
                 scheduler = None, 
                 ):
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.network = network.to(self.args.device)
        # self.network = network.to(args.device)
        
        InputTransformObj: object = InputTransforms[self.args.data]
        self.train_transform = InputTransformObj(augmentation= self.args.augmentation, randaugment= self.args.randaugment,)
        self.eval_transform = InputTransformObj(augmentation=False)
        self.train_transform.to(self.args.device)
        self.eval_transform.to(self.args.device)
        
        '''
        network = network.to(args.device)
        InputTransformObj: object = InputTransforms[args.data]
        train_transform = InputTransformObj(augmentation=args.augmentation, randaugment=args.randaugment)
        eval_transform = InputTransformObj(augmentation=False)
        train_transform.to(args.device)
        eval_transform.to(args.device)
        '''


    def fit(self, 
            train_loader, 
            id_valid_loader, 
            ood_valid_loader, 
            ):
        
        self.train_loader = train_loader
        self.id_valid_loader = id_valid_loader
        self.ood_valid_loader = ood_valid_loader
        
        opt = self.optimizer
        sch = self.scheduler(opt) if self.scheduler else None
        
        '''
        scheduler = None
        opt = optimizer
        sch = scheduler(opt) if scheduler else None
        '''

        ########################################
        # regression evaluation
        train_loss_traj, id_valid_loss_traj, ood_valid_loss_traj = [], [], []
        train_pearson_traj, id_valid_pearson_traj, ood_valid_pearson_traj = [], [], []
        train_mse_traj, id_valid_mse_traj, ood_valid_mse_traj = [], [], []
        train_mae_traj, id_valid_mae_traj, ood_valid_mae_traj = [], [], []
        rho_traj = []
        sigma_traj = []# regression only
        ########################################
        
        # CRITERION of MODEL SELECTION
        best_model, best_loss, best_metric = deepcopy(self.network.state_dict()), 1e10, 0.0
        if self.args.model_selection_metric=='mae' or self.args.model_selection_metric=='mse':
            print('mse or mae')
            # if args.model_selection_metric=='mae' or args.model_selection_metric=='mse':
            best_metric = 1e10
        else:
            pass
        '''
        best_model, best_loss, best_metric  = deepcopy(network.state_dict()), 1e10, 0.
        '''
        
        epsilon: float = 1e-5 #####
        normal = torch.distributions.normal.Normal(0, 1)
        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            self.network.train()
            train_loss, train_pred, train_true = 0., [], []
            ##### train_dataloader
            for b, batch in enumerate(self.train_loader):
                print('train batch: ', b, f'/ {len(train_loader)}' )
                x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))
                x = x.float().to(torch.float32)
                y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                batch_probits_f = self.network.forward_f(x).squeeze().to(self.args.device)
                batch_probits_g = self.network.forward_g(x).to(self.args.device)
                y_true = y
                y_pred = batch_probits_f
                loss = loss_regression(self, y, s, batch_probits_f, batch_probits_g)
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)
                train_loss += loss.item() / len(self.train_loader) # mean
                train_pred.append(y_pred.detach().cpu().numpy())
                train_true.append(y_true.detach().cpu().numpy())

            mse = mean_squared_error(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
            mae = mean_absolute_error(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
            pearson = pearsonr(np.concatenate(train_true),np.concatenate(train_pred))[0].round(3)
            
            train_loss_traj.append(train_loss)
            train_mse_traj.append(mse)
            train_mae_traj.append(mae)
            train_pearson_traj.append(pearson)
            # rho
            rho_traj.append(self.network.rho.data.detach().cpu().numpy())
            # sigma
            sigma_traj.append(self.network.sigma.data.detach().cpu().numpy())
            
            if sch:
                sch.step()

            '''
            epsilon: float = 1e-5 #####
            normal = torch.distributions.normal.Normal(0, 1)
            pbar = tqdm(range(args.epochs))
            for epoch in pbar:
                network.train()
                train_loss, train_pred, train_true = 0., [], []
                batch = next(iter(train_loader))
                for b, batch in enumerate(train_loader):
                    print('train batch: ', b, f'/ {len(train_loader)}' )
                    x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                    if train_transform is not None:
                        x = train_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                    network = network.to(args.device)
                    batch_probits_f = network.forward_f(x).squeeze().to(args.device)
                    batch_probits_g = network.forward_g(x).to(args.device)
                    loss = 0.
                    y_true = y
                    y_pred = batch_probits_f
                    for k in range(args.num_domains):
                        # k=0
                        s_true = s[:,k]
                        s_pred = batch_probits_g[:,k]
                        
                        rho = network.rho[k]
                        sigma = network.sigma

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
                    network.rho.data = torch.clip(network.rho.data, -0.99, 0.99)

                    train_loss += loss.item() / len(train_loader) # mean
                    train_pred.append(y_pred.detach().cpu().numpy())
                    train_true.append(y_true.detach().cpu().numpy())
                train_loss_traj.append(train_loss)
                mse = mean_squared_error(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
                mae = mean_absolute_error(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
                pearson = pearsonr(np.concatenate(train_true),np.concatenate(train_pred))[0].round(3)
                
                train_mse_traj.append(mse)
                train_mae_traj.append(mae)
                train_pearson_traj.append(pearson)
                # rho
                rho_traj.append(network.rho.data.detach().cpu().numpy())
                # sigma
                sigma_traj.append(network.sigma.data.detach().cpu().numpy())
            '''
            
            self.network.eval()
            with torch.no_grad():
                valid_loss_id, valid_pred_id, valid_true_id = 0., [], []
                ##### valid_dataloader
                for b, batch in enumerate(self.id_valid_loader):
                    print('ID valid batch: ', b, f'/ {len(id_valid_loader)}')
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item()) == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                    self.network = self.network.to(self.args.device)
                    batch_probits_f = self.network.forward_f(x).squeeze().to(self.args.device)
                    batch_probits_g = self.network.forward_g(x).to(self.args.device)
                    '''
                    for b, batch in enumerate(id_valid_loader):
                        print('ID valid batch: ', b, f'/ {len(id_valid_loader)}')
                        batch
                    print('ID valid batch: ', b, f'/ {len(id_valid_loader)}')
                    batch = batch_id
                    x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                    if eval_transform is not None:
                        x = eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                    len(col_idx)
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                    network = network.to(args.device)
                    batch_probits_f = network.forward_f(x).squeeze().to(args.device)
                    batch_probits_g = network.forward_g(x).to(args.device)
                    '''
                    
                    y_true = y
                    y_pred = batch_probits_f
                    
                    loss = loss_regression(self, y, s, batch_probits_f, batch_probits_g)
                    valid_loss_id += loss.item() / len(id_valid_loader)
                    valid_pred_id.append(y_pred.detach().cpu().numpy())
                    valid_true_id.append(y_true.detach().cpu().numpy())
                    '''
                    valid_loss_id += loss.item() / len(id_valid_loader)
                    valid_pred_id.append(y_pred.detach().cpu().numpy())
                    valid_true_id.append(y_true.detach().cpu().numpy())
                    '''
            
            print('valid_loss_id', valid_loss_id)
            mse_id = mean_squared_error(np.concatenate(valid_true_id),np.concatenate(valid_pred_id)).round(3)
            mae_id = mean_absolute_error(np.concatenate(valid_true_id),np.concatenate(valid_pred_id)).round(3)
            pearson_id = pearsonr(np.concatenate(valid_true_id),np.concatenate(valid_pred_id))[0].round(3)
            
            id_valid_loss_traj.append(valid_loss_id)
            id_valid_mse_traj.append(mse_id)
            id_valid_mae_traj.append(mae_id)
            id_valid_pearson_traj.append(pearson_id)
            
            ##### OOD validation
            self.network.eval()
            with torch.no_grad():
                valid_loss_ood, valid_pred_ood, valid_true_ood = 0., [], []
                ##### valid_dataloader
                for b, batch in enumerate(self.ood_valid_loader):
                    print('OOD valid batch: ', b, f'/ {len(ood_valid_loader)}')
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                    self.network = self.network.to(self.args.device)
                    batch_probits_f = self.network.forward_f(x).squeeze().to(self.args.device)
                    batch_probits_g = self.network.forward_g(x).to(self.args.device)
                    
                    y_true = y
                    y_pred = batch_probits_f
                    
                    loss = loss_regression(self, y, s, batch_probits_f, batch_probits_g)
                    
                    valid_loss_ood += loss.item() / len(ood_valid_loader)
                    valid_pred_ood.append(y_pred.detach().cpu().numpy())
                    valid_true_ood.append(y_true.detach().cpu().numpy())
                    '''
                    valid_loss_ood += loss / len(ood_valid_loader)
                    valid_pred_ood.append(y_pred.detach().cpu().numpy())
                    valid_true_ood.append(y_true.detach().cpu().numpy())
                    '''            
            print('valid_loss_ood', valid_loss_ood)
            mse_ood = mean_squared_error(np.concatenate(valid_true_ood),np.concatenate(valid_pred_ood)).round(3)
            mae_ood = mean_absolute_error(np.concatenate(valid_true_ood),np.concatenate(valid_pred_ood)).round(3)
            pearson_ood = pearsonr(np.concatenate(valid_true_ood),np.concatenate(valid_pred_ood))[0]#.round(3)
            
            ood_valid_loss_traj.append(valid_loss_ood)
            ood_valid_mse_traj.append(mse_ood)
            ood_valid_mae_traj.append(mae_ood)
            ood_valid_pearson_traj.append(pearson_ood)

            # model_selection: loss
            print(self.args.model_selection)
            if self.args.model_selection == 'loss':
                valid_loss = (self.args.w * valid_loss_id) + valid_loss_ood
                valid_loss = (self.args.w * valid_loss_id) + ((1-self.args.w) * valid_loss_ood)
                if valid_loss < best_loss:
                    best_model = deepcopy(self.network.state_dict())
                    best_loss = valid_loss
                    best_epoch = epoch
                else:
                    pass
            # model_selection: metric
            elif self.args.model_selection == 'metric':
                if self.args.model_selection_metric=='mse':
                    valid_metric = (self.args.w * mse_id) + mse_ood
                    valid_metric = (self.args.w * mse_id) + ((1-self.args.w) * mse_ood)
                    if valid_metric < best_metric:
                        best_model = deepcopy(self.network.state_dict())
                        best_metric = valid_metric
                        best_epoch = epoch
                    else:
                        pass
                elif self.args.model_selection_metric=='mae':
                    valid_metric = (self.args.w * mae_id) + mae_ood
                    valid_metric = (self.args.w * mae_id) + ((1-self.args.w) * mae_ood)
                    if valid_metric < best_metric:
                        best_model = deepcopy(self.network.state_dict())
                        best_metric = valid_metric
                        best_epoch = epoch
                    else:
                        pass
                elif self.args.model_selection_metric=='pearson':
                    valid_metric = (self.args.w * pearson_id) + pearson_ood
                    valid_metric = (self.args.w * pearson_id) + ((1-self.args.w) * pearson_ood)
                    if valid_metric > best_metric:
                        best_model = deepcopy(self.network.state_dict())
                        best_metric = valid_metric
                        best_epoch = epoch
                        # best_epoch = 0
                    else:
                        pass
                else:
                    print('choose mse, mae, or pearson')
            else:
                print('choose model selection type')

            print('done')
            desc = f'[{epoch + 1:03d}/{self.args.epochs}] '
            # desc = f'[{epoch + 1:03d}/{args.epochs}] '
            # Training
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            desc += f'tr_mse: {train_mse_traj[-1]:.4f} '
            desc += f'tr_mae: {train_mae_traj[-1]:.4f} '
            desc += f'tr_pearson: {train_pearson_traj[-1]:.4f} '
            # ID validation
            desc += f'| ID valid | Loss: {id_valid_loss_traj[-1]:.4f} '
            desc += f'id_val_mse: {id_valid_mse_traj[-1]:.4f} '
            desc += f'id_val_mae: {id_valid_mae_traj[-1]:.4f} '
            desc += f'id_val_pearson: {id_valid_pearson_traj[-1]:.4f} '
            # OOD validation
            desc += f'| OOD valid | Loss: {ood_valid_loss_traj[-1]:.4f} '
            desc += f'ood_val_mse: {ood_valid_mse_traj[-1]:.4f} '
            desc += f'ood_val_mae: {ood_valid_mae_traj[-1]:.4f} '
            desc += f'ood_val_pearson: {ood_valid_pearson_traj[-1]:.4f} '
            pbar.set_description(desc)


        self.network.load_state_dict(best_model)
        # Traj Loss & Metric: Training
        self.train_loss_traj = train_loss_traj
        self.train_mse_traj = train_mse_traj
        self.train_mae_traj = train_mae_traj
        self.train_pearson_traj = train_pearson_traj
        # Traj Loss & Metric: ID validation
        self.id_valid_loss_traj = id_valid_loss_traj
        self.id_valid_mse_traj = id_valid_mse_traj
        self.id_valid_mae_traj = id_valid_mae_traj
        self.id_valid_pearson_traj = id_valid_pearson_traj
        # Traj Loss & Metric: OOD validation
        self.ood_valid_loss_traj = ood_valid_loss_traj
        self.ood_valid_mse_traj = ood_valid_mse_traj
        self.ood_valid_mae_traj = ood_valid_mae_traj
        self.ood_valid_pearson_traj = ood_valid_pearson_traj
        
        if best_epoch is None:
            best_epoch = epoch
        else:
            pass
        # Rho
        self.rho_traj = rho_traj        
        self.sigma_traj = sigma_traj
        # Best losses
        self.best_rho = self.rho_traj[best_epoch]
        self.best_sigma = self.sigma_traj[best_epoch]
        # Best Loss & Metric: Training
        self.best_train_loss = self.train_loss_traj[best_epoch]
        self.best_train_mse = self.train_mse_traj[best_epoch]
        self.best_train_mae = self.train_mae_traj[best_epoch]
        self.best_train_pearson = self.train_pearson_traj[best_epoch]
        # Best Loss & Metric: Training
        self.best_id_valid_loss = self.id_valid_loss_traj[best_epoch]
        self.best_id_valid_mse = self.id_valid_mse_traj[best_epoch]
        self.best_id_valid_mae = self.id_valid_mae_traj[best_epoch]
        self.best_id_valid_pearson = self.id_valid_pearson_traj[best_epoch]
        # Best Loss & Metric: OOD Validation
        self.best_ood_valid_loss = self.ood_valid_loss_traj[best_epoch]
        self.best_ood_valid_mse = self.ood_valid_mse_traj[best_epoch]
        self.best_ood_valid_mae = self.ood_valid_mae_traj[best_epoch]
        self.best_ood_valid_pearson = self.ood_valid_pearson_traj[best_epoch]
        # Best Loss & best_epoch
        self.best_epoch = best_epoch
        if self.args.model_selection == 'loss':
            self.best_valid_metric = best_loss
        elif self.args.model_selection == 'metric':
            self.best_valid_metric = best_metric

    def predict(self, batch):
        # normal = torch.distributions.normal.Normal(0, 1)
        x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
        if self.eval_transform is not None:
            x = self.eval_transform(x.to(torch.uint8))
        x = x.float().to(torch.float32)
        with torch.no_grad():
            y_pred_batch = self.network.forward_f(x.to(self.args.device))
            # network.forward_f(x.cpu())[:,0]
        y_pred_batch = y_pred_batch.detach().cpu().numpy()
        return y_pred_batch
    
    def predict_loader(self, dataloader):
        y_pred_list=[]
        for b, batch in enumerate(dataloader):
            print('predict batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            if self.eval_transform is not None:
                x = self.eval_transform(x.to(torch.uint8))
            x = x.float().to(torch.float32)
            with torch.no_grad():
                y_pred_batch = self.network.forward_f(x.to(self.args.device))
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
