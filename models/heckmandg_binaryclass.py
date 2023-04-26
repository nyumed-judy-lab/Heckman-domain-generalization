import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils_datasets.transforms import InputTransforms


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
    The decomposition is argsbased on:
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

def loss_binary_classification(self,
                            y, 
                            s,
                            batch_probits_f, 
                            batch_probits_g,
                            ):
    safe_log = lambda x: torch.log(torch.clip(x, 1e-6))
    normal = torch.distributions.normal.Normal(0, 1)
    loss = 0.
    for k in range(self.args.num_domains):
        # k=0
        # joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits_g[:, k], self.network.rho[k])
        joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], self.network.rho[k])
        joint_prob = bivariate_normal_cdf(batch_probits_g[:, k], batch_probits_f, self.network.rho[k])
        # joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
        loss += -(y * s[:, k] * safe_log(joint_prob)).mean() # equation (22) - 1
        loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean() # equation (22) - 2
        loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean() # equation (22) - 3
    return loss    


def _cross_domain_binary_classification_loss(self,
                                                y_pred: torch.FloatTensor,
                                                y_true: torch.LongTensor,
                                                s_pred: torch.FloatTensor,
                                                s_true: torch.LongTensor,
                                                rho   : torch.FloatTensor, ) -> torch.FloatTensor:
    # This loss function (was used in ICLR paper) doesn't work in ood validation.
    """
    Arguments:
        y_pred : 1d `torch.FloatTensor` of shape (N,  ); in probits.
        y_true : 1d `torch.LongTensor`  of shape (N,  ); with values in {0, 1}.
        s_pred : 2d `torch.FloatTensor` of shape (B, K); in probits.
        s_true : 1d `torch.LongTensor`  of shape (N,  ); with values in [0, K-1].
        rho    : 1d `torch.FloatTensor` of shape (N,  ); with values in [-1, 1].
    Returns:
        ...
    """

    _epsilon: float = 1e-7
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    # Gather from `s_pred` values with indices that correspond to the true domains
    s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()  # (N,  )

    # - log Pr[S_k = 1, Y = 1]; shape = (N,  )
    loss_selected_pos = - y_true.float() * torch.log(
        self._bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + _epsilon,
    )
    loss_selected_pos = torch.nan_to_num(loss_selected_pos, nan=0., posinf=0., neginf=0.)
    loss_selected_pos = loss_selected_pos[y_true.bool()]

    # - log Pr[S_k = 1, Y = 0]; shape = (N,  )
    loss_selected_neg = - (1 - y_true.float()) * torch.log(
        _normal.cdf(s_pred_k) - self._bivariate_normal_cdf(a=s_pred_k, b=y_pred, rho=rho) + _epsilon
    )
    loss_selected_neg = torch.nan_to_num(loss_selected_neg, nan=0., posinf=0., neginf=0.)
    loss_selected_neg = loss_selected_neg[(1 - y_true).bool()]

    loss_selected = torch.cat([loss_selected_pos, loss_selected_neg], dim=0)

    # Create a 2d indicator for `s_true`
    #   Shape; (N, K)
    s_true_2d = torch.zeros_like(s_pred).scatter_(
        dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred)
    )

    # -\log Pr[S_l = 0] for l \neq k
    loss_not_selected = - torch.log(1 - _normal.cdf(s_pred) + _epsilon)     # (N, K)
    loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N, K)
    loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (NK - N,  )
    
    return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()

    # total_count: int = loss_selected.numel() + loss_not_selected.numel()
    # sel_weight: float = total_count / len(loss_selected)
    # not_sel_weight: float = total_count / len(loss_not_selected)
    # return torch.cat([loss_selected * sel_weight, loss_not_selected * not_sel_weight], dim=0).mean()

class HeckmanDG_CNN_BinaryClassifier:
    def __init__(self, args,
                 network, 
                 optimizer, 
                 scheduler = None, 
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
        # Classification evaluation
        train_loss_traj, id_valid_loss_traj, ood_valid_loss_traj = [], [], []
        train_auc_traj, id_valid_auc_traj, ood_valid_auc_traj = [], [], []
        train_acc_traj, id_valid_acc_traj, ood_valid_acc_traj = [], [], []
        train_f1_traj, id_valid_f1_traj, ood_valid_f1_traj = [], [], []
        rho_traj = []
        ########################################
        
        # CRITERION of MODEL SELECTION
        best_model, best_loss, best_metric  = deepcopy(self.network.state_dict()), 1e10, 0.
        
        '''
        best_model, best_loss, best_metric  = deepcopy(network.state_dict()), 1e10, 0.
        '''
        
        normal = torch.distributions.normal.Normal(0, 1)
        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            # epoch = 0
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
                
                # predict
                batch_probits_f = self.network.forward_f(x).squeeze()
                batch_probits_g = self.network.forward_g(x)
                batch_prob = normal.cdf(batch_probits_f) # y_pred
                
                loss = loss_binary_classification(self, y=y, s=s, batch_probits_f = batch_probits_f, batch_probits_g=batch_probits_g)
                # input: in probits
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)
                train_loss += loss.item() / len(self.train_loader)
                train_pred.append(batch_prob.detach().cpu().numpy())
                train_true.append(y.detach().cpu().numpy())
                
            train_loss_traj.append(train_loss)
            auc = roc_auc_score(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
            acc = accuracy_score(np.concatenate(train_true),(np.concatenate(train_pred)>0.5).astype(int)).round(3)
            f1 = f1_score(np.concatenate(train_true),(np.concatenate(train_pred)>0.5).astype(int),  average='macro').round(3)
            train_auc_traj.append(auc)
            train_acc_traj.append(acc)
            train_f1_traj.append(f1)
            
            # rho
            rho_traj.append(self.network.rho.data.detach().cpu().numpy())
            # scheduler
            if sch:
                sch.step()
            '''
            normal = torch.distributions.normal.Normal(0, 1)
            pbar = tqdm(range(args.epochs))    
            for epoch in pbar:
                # epoch = 0
                network.train()
                train_loss = 0.
                train_pred, train_true = [], []
                normal = torch.distributions.normal.Normal(0, 1)
                pbar = tqdm(range(args.epochs))
                for b, batch in enumerate(train_loader):
                    print('train batch: ', b, f'/ {len(train_loader)}' )
                    x = batch['x'].to(args.device).to(torch.float32)
                    x = train_transform(x.to(torch.uint8))
                    y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                    batch_probits_f = network.forward_f(x).squeeze()
                    batch_probits_g = network.forward_g(x)
                    batch_prob = normal.cdf(batch_probits_f)
                    loss = 0.
                    for k in range(args.num_domains):
                        # k=0
                        joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
                        loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
                        loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                        loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    network.rho.data = torch.clip(network.rho.data, -0.99, 0.99)
                    train_loss += loss.item() / len(train_loader)
                    train_pred.append(batch_prob.detach().cpu().numpy())
                    train_true.append(y.detach().cpu().numpy())
                train_loss_traj.append(train_loss)
                auc = roc_auc_score(np.concatenate(train_true),np.concatenate(train_pred)).round(3)
                acc = accuracy_score(np.concatenate(train_true),(np.concatenate(train_pred)>0.5).astype(int)).round(3)
                f1 = f1_score(np.concatenate(train_true),(np.concatenate(train_pred)>0.5).astype(int),  average='macro').round(3)
                train_auc_traj.append(auc)
                train_acc_traj.append(acc)
                train_f1_traj.append(f1)
                
                # rho
                rho_traj.append(network.rho.data.detach().cpu().numpy())
            '''
            ##### ID Validation
            self.network.eval()
            with torch.no_grad():
                valid_loss_id, valid_pred_id, valid_true_id = 0., [], []
                for b, batch in enumerate(self.id_valid_loader):
                    print('id_valid_loader batch: ', b, f'/ {len(self.id_valid_loader)}' )
                    # batch preprocessing: x, y, s
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                    # predict
                    batch_probits_f = self.network.forward_f(x).squeeze()
                    batch_probits_g = self.network.forward_g(x)
                    batch_prob = normal.cdf(batch_probits_f)

                    loss = loss_binary_classification(self, y=y, s=s, batch_probits_f = batch_probits_f, batch_probits_g=batch_probits_g)

                    valid_loss_id += loss.item() / len(id_valid_loader)
                    valid_pred_id.append(batch_prob.detach().cpu().numpy())
                    valid_true_id.append(y.detach().cpu().numpy())

                    '''
                    network.eval()
                    with torch.no_grad():
                        valid_loss_id, valid_pred_id, valid_true_id = 0., [], []
                        for b, batch in enumerate(id_valid_loader):
                            print('id_valid_loader batch: ', b, f'/ {len(id_valid_loader)}' )
                            # batch preprocessing: x, y, s
                            x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                            x = eval_transform(x.to(torch.uint8))
                            y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                            one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                            col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                            for i in range(len(one_hot)):
                                one_hot[i,col_idx[i]] = 1
                            s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                            # predict
                            batch_probits_f = network.forward_f(x).squeeze()
                            batch_probits_g = network.forward_g(x)
                            batch_prob = normal.cdf(batch_probits_f)
                            loss = 0.
                            for k in range(args.num_domains):
                                # k=0
                                joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
                                loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
                                loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                                loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()
                            valid_loss_id += loss.item() / len(id_valid_loader)
                            valid_pred_id.append(batch_prob.detach().cpu().numpy())
                            valid_true_id.append(y.detach().cpu().numpy())
                    valid_loss_id += loss.item() / len(id_valid_loader)
                    valid_pred_id.append(batch_prob.detach().cpu().numpy())
                    valid_true_id.append(y.detach().cpu().numpy())
                    '''
            
            print('valid_loss_id', valid_loss_id)
            auc_id = roc_auc_score(np.concatenate(valid_true_id),np.concatenate(valid_pred_id)).round(3)
            acc_id = accuracy_score(np.concatenate(valid_true_id),(np.concatenate(valid_pred_id)>0.5).astype(int)).round(3)
            f1_id = f1_score(np.concatenate(valid_true_id),(np.concatenate(valid_pred_id)>0.5).astype(int),  average='macro').round(3)
            id_valid_loss_traj.append(valid_loss_id)
            id_valid_auc_traj.append(auc_id)
            id_valid_acc_traj.append(acc_id)
            id_valid_f1_traj.append(f1_id)

            ##### OOD Validation
            self.network.eval()
            with torch.no_grad():
                valid_loss_ood, valid_pred_ood, valid_true_ood = 0.0, [], []
                for b, batch in enumerate(self.ood_valid_loader):
                    print('ood_valid_loader batch: ', b, f'/ {len(self.ood_valid_loader)}' )
                    
                    # batch preprocessing
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    y = batch['y'].to(self.args.device).to(torch.float32)  # (B, *)
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B, *)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, *)
                    
                    # predict
                    batch_probits_f = self.network.forward_f(x).squeeze()
                    batch_probits_g = self.network.forward_g(x)
                    batch_prob = normal.cdf(batch_probits_f)
                    
                    loss = loss_binary_classification(self, y=y, s=s, batch_probits_f = batch_probits_f, batch_probits_g=batch_probits_g)

                    valid_loss_ood += loss.item() / len(ood_valid_loader)
                    valid_pred_ood.append(batch_prob.detach().cpu().numpy())
                    valid_true_ood.append(y.detach().cpu().numpy())            
                    '''
                    network.eval()
                    with torch.no_grad():
                        valid_loss_ood, valid_pred_ood, valid_true_ood = 0.0, [], []
                        for b, batch in enumerate(ood_valid_loader):
                            print('ood_valid_loader: ', b, f'/ {len(ood_valid_loader)}' )
                            x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                            x = eval_transform(x.to(torch.uint8))
                            y = batch['y'].to(args.device).to(torch.float32)  # (B, *)
                            s_ = batch['domain'].to(args.device).to(torch.float32)  # (B, *)
                            one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                            s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, *)
                                    
                            batch_probits_f = network.forward_f(x).squeeze()
                            batch_probits_g = network.forward_g(x)
                            batch_prob = normal.cdf(batch_probits_f)
                            
                            loss = 0.
                            for k in range(args.num_domains):
                                # k=0
                                joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
                                loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
                                loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
                                loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()
                            valid_loss_ood += loss.item() / len(ood_valid_loader)
                            valid_pred_ood.append(batch_prob.detach().cpu().numpy())
                            valid_true_ood.append(y.detach().cpu().numpy())            
                    '''
            
            print('valid_loss_ood', valid_loss_ood)
            auc_ood = roc_auc_score(np.concatenate(valid_true_ood),np.concatenate(valid_pred_ood)).round(3)
            acc_ood = accuracy_score(np.concatenate(valid_true_ood),(np.concatenate(valid_pred_ood)>0.5).astype(int)).round(3)
            f1_ood = f1_score(np.concatenate(valid_true_ood),(np.concatenate(valid_pred_ood)>0.5).astype(int),  average='macro').round(3)
            ood_valid_loss_traj.append(valid_loss_ood)
            ood_valid_auc_traj.append(auc_ood)
            ood_valid_acc_traj.append(acc_ood)
            ood_valid_f1_traj.append(f1_ood)
            
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
                if self.args.model_selection_metric=='auc':
                    valid_metric = (self.args.w * auc_id) + auc_ood
                    valid_metric = (self.args.w * auc_id) + ((1-self.args.w) * auc_ood)
                elif self.args.model_selection_metric=='f1':
                    valid_metric = (self.args.w * f1_id) + f1_ood
                    valid_metric = (self.args.w * f1_id) + ((1-self.args.w) * f1_ood)
                elif self.args.model_selection_metric=='accuracy':
                    valid_metric = (self.args.w * acc_id) + acc_ood
                    valid_metric = (self.args.w * acc_id) + ((1-self.args.w) * acc_ood)
                else:
                    print('choose auc, f1, or accuracy')
                if valid_metric > best_metric:
                    best_model = deepcopy(self.network.state_dict())
                    best_metric = valid_metric
                    best_epoch = epoch
                else:
                    pass
            else:
                print('choose model selection type')
            '''
            # model_selection: loss
            if args.model_selection == 'loss':
                valid_loss = (args.w * valid_loss_id) + ((1-args.w) * valid_loss_ood)
            # model_selection: metric
            elif args.model_selection == 'metric':
                if args.model_selection_metric=='auc':
                    valid_metric = (args.w * auc_id) + ((1-args.w) * auc_ood)
                elif args.model_selection_metric=='f1':
                    valid_metric = (args.w * f1_id) + ((1-args.w) * f1_ood)
                elif args.model_selection_metric=='accuracy':
                    valid_metric = (args.w * acc_id) + ((1-args.w) * acc_ood)
                else:
                    print('choose auc, f1, or accuracy')
            # model_selection: loss
            if args.model_selection == 'loss':
                if valid_loss < best_loss:
                    best_model = deepcopy(network.state_dict())
                    best_loss = valid_loss
                    best_epoch = epoch
                else:
                    pass
            # model_selection: metric
            elif args.model_selection == 'metric':
                if valid_metric > best_metric:
                    best_model = deepcopy(network.state_dict())
                    best_metric = valid_metric
                    best_epoch = epoch
                    best_epoch = epoch
                else:
                    pass
            else:
                print('choose model selection type')
            '''

            print('done')
            desc = f'[{epoch + 1:03d}/{self.args.epochs}] '
            # desc = f'[{epoch + 1:03d}/{args.epochs}] '
            # Training
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            desc += f'tr_auc: {train_auc_traj[-1]:.4f} '
            desc += f'tr_acc: {train_acc_traj[-1]:.4f} '
            desc += f'tr_f1: {train_f1_traj[-1]:.4f} '
            # ID validation
            desc += f'| ID valid | Loss: {id_valid_loss_traj[-1]:.4f} '
            desc += f'id_val_auc: {id_valid_auc_traj[-1]:.4f} '
            desc += f'id_val_acc: {id_valid_acc_traj[-1]:.4f} '
            desc += f'id_val_f1: {id_valid_f1_traj[-1]:.4f} '
            # OOD validation
            desc += f'| OOD valid | Loss: {ood_valid_loss_traj[-1]:.4f} '
            desc += f'ood_val_auc: {ood_valid_auc_traj[-1]:.4f} '
            desc += f'ood_val_acc: {ood_valid_acc_traj[-1]:.4f} '
            desc += f'ood_val_f1: {ood_valid_f1_traj[-1]:.4f} '
            pbar.set_description(desc)

        # selected model
        self.network.load_state_dict(best_model)
        
        # Traj Loss & Metric: Training
        self.train_loss_traj = train_loss_traj
        self.train_acc_traj = train_acc_traj
        self.train_auc_traj = train_auc_traj
        self.train_f1_traj = train_f1_traj
        # Traj Loss & Metric: ID validation
        self.id_valid_loss_traj = id_valid_loss_traj
        self.id_valid_auc_traj = id_valid_auc_traj
        self.id_valid_acc_traj = id_valid_acc_traj
        self.id_valid_f1_traj = id_valid_f1_traj
        # Traj Loss & Metric: OOD validation
        self.ood_valid_loss_traj = ood_valid_loss_traj
        self.ood_valid_auc_traj = ood_valid_auc_traj
        self.ood_valid_acc_traj = ood_valid_acc_traj
        self.ood_valid_f1_traj = ood_valid_f1_traj
        # Rho
        self.rho_traj = rho_traj        
        
        if best_epoch is None:
            best_epoch = epoch
        else:
            pass
        
        # Best losses
        self.best_rho = rho_traj[best_epoch]
        # Best Loss & Metric: Training
        self.best_train_loss = self.train_loss_traj[best_epoch]
        self.best_train_auc = self.train_auc_traj[best_epoch]
        self.best_train_acc = self.train_acc_traj[best_epoch]
        self.best_train_f1 = self.train_f1_traj[best_epoch]
        # Best Loss & Metric: Training
        self.best_id_valid_loss = self.id_valid_loss_traj[best_epoch]
        self.best_id_valid_auc = self.id_valid_auc_traj[best_epoch]
        self.best_id_valid_acc = self.id_valid_acc_traj[best_epoch]
        self.best_id_valid_f1 = self.id_valid_f1_traj[best_epoch]
        # Best Loss & Metric: OOD Validation
        self.best_ood_valid_loss = self.ood_valid_loss_traj[best_epoch]
        self.best_ood_valid_auc = self.ood_valid_auc_traj[best_epoch]
        self.best_ood_valid_acc = self.ood_valid_acc_traj[best_epoch]
        self.best_ood_valid_f1 = self.ood_valid_f1_traj[best_epoch]
        # Best Loss & best_epoch
        self.best_epoch = best_epoch
        if self.args.model_selection == 'loss':
            self.best_valid_metric = best_loss
        elif self.args.model_selection == 'metric':
            self.best_valid_metric = best_metric

    def predict_proba(self, batch):
        normal = torch.distributions.normal.Normal(0, 1)
        x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
        if self.eval_transform is not None:
            x = self.eval_transform(x.to(torch.uint8))
        x = x.float().to(torch.float32)
        with torch.no_grad():
            probs_batch = normal.cdf(self.network.forward_f(x.to(self.args.device)))
        probs_batch = probs_batch.detach().cpu().numpy()
        return probs_batch
    
    def predict_proba_loader(self, dataloader):
        normal = torch.distributions.normal.Normal(0, 1)
        probs_list=[]
        for b, batch in enumerate(dataloader):
            print('predict batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            if self.eval_transform is not None:
                x = self.eval_transform(x.to(torch.uint8))
            x = x.float().to(torch.float32)
            with torch.no_grad():
                probs_batch = normal.cdf(self.network.forward_f(x.to(self.args.device)))
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
    

class HeckmanDG_DNN_BinaryClassifier:
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