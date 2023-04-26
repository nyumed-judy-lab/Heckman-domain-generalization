import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import typing
from networks.cnn_initializers import NetworkInitializer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils_datasets.transforms import InputTransforms

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))

from utils.multinomial_utils import MatrixOps, truncnorm_rvs_recursive

def multiclass_classification_loss(self,
                                   y_pred: torch.FloatTensor,    # shape: (B, K)
                                   y_true: torch.LongTensor,     # shape: (B)
                                   s_pred: torch.FloatTensor,    # shape: (B, K)
                                   s_true: torch.LongTensor,     # shape: (B, 1)
                                   rho: torch.FloatTensor,       # shape: (K, J+1)
                                   approximate: bool = False,    # logistic approx.
                                   **kwargs, ) -> torch.FloatTensor:
    # args = self.args
    # args.device = 'cpu'
    _eps: float = 1e-7
    _normal = torch.distributions.Normal(loc=0., scale=1.)
    _float_type = y_pred.dtype  # creating new tensors (in case of amp)
    
    # loss with logistic approximation available.
    # B = int(y_true.size(0))  # batch size
    # J = int(y_pred.size(1))  # number of classes
    # K = int(s_pred.size(1))  # number of (training) domains

    B = y_true.shape[0]  # batch size
    J = y_pred.shape[1]  # number of classes
    K = s_pred.shape[1]  # number of (training) domains

    s_idx = [np.where(int(s_true[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_true))]
    # idx = 0
    # s_idx = [np.where(int(s_true[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_true))]
    s_ind_ = torch.Tensor(s_idx).to(int)
    s_index = s_ind_.unsqueeze(-1) #.shape
    s_index = s_index.to(self.args.device)
    s_pred_k = s_pred.gather(dim=1, index=s_index) # s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).flatten()
    rho_k = rho[s_index.squeeze()] # rho_s.shape # (B of K, J+1)
    assert len(s_pred_k) == len(s_pred)
    
    # s_pred_k = s_pred.gather(dim=1, index=s_true.to(int))
    # s_pred_k_ = torch.zeros((B, 1))
    # for b in range(B):
    #     s_pred_k_[b] = s_pred[b,s_index[b]]
    # approximate=True
    assert s_true.shape == (B, 1)
    assert s_pred_k.shape == (B, 1)
    assert s_pred.shape ==  (B, K)
    assert y_true.shape ==  (B,)
    assert y_pred.shape == (B, J)
    assert rho.shape == (K, J+1)
    assert rho_k.shape == (B, J+1)
    
    # matrix of y (probit) differences (with respect to its true outcome)
    y_pred_diff = torch.zeros(B, J-1, dtype=_float_type, device=self.args.device)
    
    for j in range(J):
        ## j=0
        col_mask = torch.arange(0, J, device=self.args.device).not_equal(j)
        row_mask = y_true.eq(j)
        y_pred_diff[row_mask, :] = (y_pred[:, j].view(-1, 1) - y_pred[:, col_mask])[row_mask, :]
    
    assert len(rho_k) == len(y_pred)
    assert rho_k.shape[1] == (y_pred.shape[1] + 1)

    C_tilde_list = list()
    for i in range(B):
        # i=0
        L = torch.eye(J + 1, device=rho_k.device)  # construct a lower triangular matrix L.shape
        L[-1] = rho_k[i]                           # fill in params
        C = MatrixOps.cov_to_corr(L @ L.T)       # (J+1, J+1) == C.shape
        j = int(y_true[i].item())                # true target index
        Cy = C[:J, :J].clone()                   # (J, J)  == Cy.shape
        Cy_diff = MatrixOps.compute_cov_of_error_differences(Cy, j=j)  # (J-1, J-1)
        C_tilde = torch.empty(J, J, device=rho_k.device)                 # (J, J)
        C_tilde[:J-1, :J-1] = Cy_diff
        not_j = torch.arange(0, J, device=rho_k.device).not_equal(j)
        not_j = not_j.nonzero(as_tuple=True)[0]
        C_tilde[-1, :-1] = C[-1, j] - C[-1, not_j]                     # (1,  ) - (1, J-1)
        C_tilde[:-1, -1] = C[j, -1] - C[not_j, -1]                     # ...
        C_tilde[-1, -1] = C[-1, -1]                                    # equals 1
        C_tilde = MatrixOps.make_positive_definite(C_tilde)            # not necessary
        C_tilde_list += [C_tilde]

    # Cholesky decomposition; (B, J, J)
    L = torch.linalg.cholesky(torch.stack(C_tilde_list, dim=0))
    L_lower = L - torch.diag_embed(
        torch.diagonal(L, offset=0, dim1=1, dim2=2),
        dim1=1, dim2=2
    )
    # print(L.shape, L.shape)
    # print(L_lower.shape, L_lower.shape)
    
    # GHK algorithm
    _probs = torch.ones(B, J, dtype=_float_type, device=self.args.device)
    # _probs = torch.ones(B, J, dtype=_float_type, device=args.device)
    v = torch.zeros_like(_probs)
    for l in range(J):         # 0, 1, ..., J-2, J-1
        if l < (J - 1):
            # l=0
            torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2).shape
            y_pred_diff.shape
            lower_trunc = - (  # (B,  )
                y_pred_diff[:, l] - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]
            ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )
            a = - (y_pred_diff[:, l] - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            b = torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l]
            a.div(b).shape
            lower_trunc.shape
        else:
            # l=12
            lower_trunc = - (  # (B,  )
                s_pred_k - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]  # (B,  )
            ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )
            
            a = - (s_pred_k - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            a = (s_pred_k.squeeze() - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            b = torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l]
            lower_trunc = -a.div(b)

        # sample from truncated normal (in batches)
        lower_trunc_numpy = lower_trunc.detach().cpu().numpy()
        # lower_trunc_numpy.shape
        samples = truncnorm_rvs_recursive(
            loc=np.zeros(B), scale=np.ones(B),
            lower_clip=lower_trunc_numpy, max_iter=5,
        )

        v[:, l] = torch.from_numpy(samples).to(self.args.device).flatten()
        # v[:, l] = torch.from_numpy(samples).to(args.device).flatten()
        _probs[:, l] = 1. - _normal.cdf(lower_trunc)

    if approximate:
        # y_probs_all = F.softmax(y_pred, dim=1)
        # assert y_probs_all.shape == (B, J)
        y_probs = F.softmax(y_pred, dim=1).gather(dim=1, index=y_true.view(-1, 1).to(int)).flatten()
    else:
        y_probs = torch.prod(_probs[:, :-1], dim=1).flatten()

    # Pr[Y = j, S_k = 1 | X]
    y_s_joint_probs = y_probs * _probs[:, -1].flatten()  # the last column of `_probs` gives Pr[S_k = 1 | Y = j, X]
    loss_selected = - torch.log(y_s_joint_probs + _eps)

    # Pr[S_l = 0 | X], for l \neq k
    s_true_2d = torch.zeros_like(s_pred).scatter_(
        dim=1, index=s_index.view(-1, 1).to(int), src=torch.ones_like(s_pred),
    )
    
    loss_not_selected = - torch.log(1. - _normal.cdf(s_pred) + _eps)
    loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.,)
    loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())  # s_true_2d = 0

    return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()


def _cross_domain_multiclass_classification_loss(#self,
                                                    y_pred: torch.FloatTensor,
                                                    y_true: torch.LongTensor,
                                                    s_pred: torch.FloatTensor,
                                                    s_true: torch.LongTensor,
                                                    rho: torch.FloatTensor,       # shape; (N, J+1)
                                                    approximate: bool = False,    # logistic approx.
                                                    **kwargs, ) -> torch.FloatTensor:
    """Multinomial loss with logistic approximation available."""
    '''
    y_pred=probits_or_resp,
    y_true=target
    s_pred=s_pred_in_probits
    s_true=s_true_1d
    rho=rho
    approximate=True
    '''
    _eps: float = 1e-7
    _normal = torch.distributions.Normal(loc=0., scale=1.)
    _float_type = y_pred.dtype  # creating new tensors (in case of amp)

    B = int(y_true.size(0))  # batch size
    J = int(y_pred.size(1))  # number of classes
    K = int(s_pred.size(1))  # number of (training) domains
    s_pred_k = s_pred.gather(dim=1, index=s_true.to(int))
    
    assert len(s_pred_k) == len(s_pred)

    # matrix of y(probit) differences (with respect to its true outcome)
    # y_pred_diff = torch.zeros(B, J-1, dtype=_float_type, device=self.device)
    y_pred_diff = torch.zeros(B, J-1, dtype=_float_type, device=device)
    
    for j in range(J):
        ## j=0
        # col_mask = torch.arange(0, J, device=self.device).not_equal(j)
        col_mask = torch.arange(0, J, device=device).not_equal(j) ## 
        row_mask = y_true.eq(j)
        y_pred_diff[row_mask, :] = (y_pred[:, j].view(-1, 1) - y_pred[:, col_mask])[row_mask, :]
    assert len(rho) == len(y_pred)
    assert rho.shape[1] == (y_pred.shape[1] + 1)

    C_tilde_list = list()
    for i in range(B):
        L = torch.eye(J + 1, device=rho.device)  # construct a lower triangular matrix
        L[-1] = rho[i]                           # fill in params
        C = MatrixOps.cov_to_corr(L @ L.T)       # (J+1, J+1)
        j: int = y_true[i].item()                # true target index
        Cy = C[:J, :J].clone()                   # (J, J)
        Cy_diff = MatrixOps.compute_cov_of_error_differences(Cy, j=j)  # (J-1, J-1)
        C_tilde = torch.empty(J, J, device=rho.device)                 # (J, J)
        C_tilde[:J-1, :J-1] = Cy_diff
        not_j = torch.arange(0, J, device=rho.device).not_equal(j)
        not_j = not_j.nonzero(as_tuple=True)[0]
        C_tilde[-1, :-1] = C[-1, j] - C[-1, not_j]                     # (1,  ) - (1, J-1)
        C_tilde[:-1, -1] = C[j, -1] - C[not_j, -1]                     # ...
        C_tilde[-1, -1] = C[-1, -1]                                    # equals 1
        C_tilde = MatrixOps.make_positive_definite(C_tilde)            # not necessary
        C_tilde_list += [C_tilde]

    # Cholesky decomposition; (B, J, J)
    L = torch.linalg.cholesky(torch.stack(C_tilde_list, dim=0))
    L_lower = L - torch.diag_embed(
        torch.diagonal(L, offset=0, dim1=1, dim2=2),
        dim1=1, dim2=2
    )
    print(L.shape, L.shape)
    print(L_lower.shape, L_lower.shape)
    
 
    def truncnorm_rvs_recursive(loc, scale, lower_clip, max_iter: int = 10):
        """Add function docstring."""
        n: int = len(loc)
        q = np.random.normal(loc, scale, size=(n, ))
        mask = (q < lower_clip)  # True if not valid sample of truncated normal
        if np.any(mask):
            if max_iter > 0:
                # recursively sample
                q[mask] = truncnorm_rvs_recursive(
                    loc=loc[mask],
                    scale=scale[mask],
                    lower_clip=lower_clip[mask],
                    max_iter=max_iter-1,
                )
            else:
                q[mask] = lower_clip[mask] + 1e-5
        return q

    # GHK algorithm
    # _probs = torch.ones(B, J, dtype=_float_type, device=self.device)
    _probs = torch.ones(B, J, dtype=_float_type, device=device)
    v = torch.zeros_like(_probs)
    for l in range(J):         # 0, 1, ..., J-2, J-1
        if l < (J - 1):
            # l=0
            torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2).shape
            y_pred_diff.shape
            lower_trunc = - (  # (B,  )
                y_pred_diff[:, l] - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]
            ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )
            a = - (y_pred_diff[:, l] - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            b = torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l]
            a.div(b).shape
            lower_trunc.shape
        else:
            # l=12
            lower_trunc = - (  # (B,  )
                s_pred_k - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l]  # (B,  )
            ).div(torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l])                    # (B,  )
            
            a = - (s_pred_k - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            a = (s_pred_k.squeeze() - torch.bmm(L_lower, v.clone().unsqueeze(2)).squeeze(2)[:, l])
            b = torch.diagonal(L, offset=0, dim1=1, dim2=2)[:, l]
            lower_trunc = -a.div(b)

        # TODO: by far this is the fastest implementation, but can we do better?
        # sample from truncated normal (in batches)
        lower_trunc_numpy = lower_trunc.detach().cpu().numpy()
        # lower_trunc_numpy.shape
        samples = truncnorm_rvs_recursive(
            loc=np.zeros(B), scale=np.ones(B),
            lower_clip=lower_trunc_numpy, max_iter=5,
        )
   
        # v[:, l] = torch.from_numpy(samples).to(self.device).flatten()
        v[:, l] = torch.from_numpy(samples).to(device).flatten()
        _probs[:, l] = 1. - _normal.cdf(lower_trunc)

    if approximate:
        y_probs = F.softmax(y_pred, dim=1).gather(dim=1, index=y_true.view(-1, 1)).flatten()
    else:
        y_probs = torch.prod(_probs[:, :-1], dim=1).flatten()

    # Pr[Y = j, S_k = 1 | X]
    y_s_joint_probs = y_probs * _probs[:, -1].flatten()  # the last column of `_probs` gives Pr[S_k = 1 | Y = j, X]
    loss_selected = - torch.log(y_s_joint_probs + _eps)

    # Pr[S_l = 0 | X], for l \neq k
    s_true_2d = torch.zeros_like(s_pred).scatter_(
        dim=1, index=s_true.view(-1, 1), src=torch.ones_like(s_pred),
    )
    loss_not_selected = - torch.log(1. - _normal.cdf(s_pred) + _eps)
    loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.,)
    loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())  # s_true_2d = 0

    return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()


class HeckmanDG_CNN_MultiClassifier:
    def __init__(self, 
                 args,
                 network, 
                 optimizer, 
                 scheduler=None, 
                 ):
        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.network = network.to(self.args.device)
        
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
            self.network.train() 
            train_loss, train_pred, train_true = 0., [], []
            ##### train_dataloader
            for b, batch in enumerate(self.train_loader):
                print('train batch: ', b, f'/ {len(self.train_loader)}' )
                x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                if self.train_transform is not None:
                    x = self.train_transform(x.to(torch.uint8))
                x = x.float().to(torch.float32)
                # y
                y_ = batch['y'].to(self.args.device).to(torch.float32)  # (B)                
                one_hot = np.zeros((self.args.batch_size, self.args.num_classes))
                for i in range(len(one_hot)):
                    one_hot[i,int(y_[i])] = 1
                y = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                # s
                s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B)                
                one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                batch_probits_f = self.network.forward_f(x)
                batch_probits_g = self.network.forward_g(x)
                rho = self.network.rho # (K, J+1)
                
                '''
                normal = torch.distributions.normal.Normal(0, 1)
                pbar = tqdm(range(args.epochs))
                epoch = 0
                network.train() 
                for b, batch in enumerate(train_loader):
                    print('train batch: ', b, f'/ {len(train_loader)}' )
                    y_ = batch['y'].to(args.device).to(torch.float32)  # (B)  - CUDA
                    assert (y_ > args.num_classes).sum().item() == 0
                    
                x = batch['x'].to(args.device).to(torch.float32)#  # (B, *)
                x = train_transform(x.to(torch.uint8))
                x = x.float().to(torch.float32)
                y_ = batch['y'].to(args.device).to(torch.float32)  # (B)  - CUDA
                one_hot = np.zeros((args.batch_size, args.num_classes))
                for i in range(len(one_hot)):
                    one_hot[i,int(y_[i])] = 1
                y = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (Batch_size, # Nomains)
                s_ = batch['domain'].to(args.device).to(torch.float32)  # (Batch_size)
                one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                for i in range(len(one_hot)):
                    one_hot[i,col_idx[i]] = 1
                s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, K) s.shape s.sum()
                batch_probits_f = network.forward_f(x)
                batch_probits_g = network.forward_g(x)
                rho = network.rho # rho.shape == (K, J+1)
                
                
                s_idx = list(range(len(args.train_domains)))
                '''

                if False:
                    # this is perforomed in the loss function
                    batch_prob_f = F.softmax(batch_probits_f, dim=1) #[0,:].sum()
                
                y_true = y_.clone()                     # y_true.shape == (B,)
                s_true = s_.unsqueeze(-1).clone()       # s_true.shape == (B, 1)
                
                y_true = y.clone()                      # y_true.shape == (B, J) 
                s_true = s.clone()                      # s_true.shape == (B, K)
                y_pred = batch_probits_f.clone()        # y_pred.shape == (B, J) in probits
                s_pred = batch_probits_g.clone()        # s_pred.shape == (B, K)
                
                assert rho.shape[1] == batch_probits_f.shape[1]+1
                # loss = 0.
                loss, loss_sel, loss_not_sel = multiclass_classification_loss(self,
                                                                              y_pred=y_pred, # in probits
                                                                              y_true=y_true, 
                                                                              s_pred=s_pred, # in probits
                                                                              s_true=s_true, 
                                                                              rho=rho,
                                                                              approximate=True,)
                

                '''
                ##### current variables
                y_.shape (B) -> y_true : as the input in the loss functin
                s_.shape (B) -> s_.unsqueeze(-1) -> s_true : as the input in the loss functin
                batch_prob_f #  (B, J) -> y_pred : as the input in the loss functin
                batch_probits_g #  (B, K) -> s_pred : as the input in the loss functin
                y.shape (B, J)
                s.shape (B, K)
                ##### input of loss function
                y_true.shape #  (B)
                s_true.shape #  (B, 1)
                y_pred.shape #  (B, J)
                s_pred.shape #  (B, K)
                rho.shape # (B, J+1)
                approximate=True
                '''
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                self.network.rho.data = torch.clip(self.network.rho.data, -0.99, 0.99)
                train_loss += loss.item() / len(self.train_loader)
                train_pred.append(F.softmax(y_pred, dim=1).detach().cpu().numpy())
                train_true.append(y.detach().cpu().numpy())

            
            ######### train_pred
            y_true_set = np.concatenate(train_true)
            y_pred_set = np.concatenate(train_pred)
            idx = y_pred_set.argmax(axis=1)
            
            y_pred_class = np.zeros((len(y_pred_set), self.args.num_classes)).astype(int)
            # y_pred_class = torch.zeros(len(y_pred_set), self.args.num_classes, dtype=torch.int)
            for i in range(len(y_pred_class)):
                y_pred_class[i, idx[i]] = 1
            ######### train_loss, acc, f1
            print('train_loss', train_loss)
            acc = accuracy_score(y_true_set, y_pred_class).round(3) 
            f1 = f1_score(y_true_set, y_pred_class, average='macro', zero_division=0).round(3) ###
            ######### save
            train_loss_traj.append(train_loss)
            train_acc_traj.append(acc)
            train_f1_traj.append(f1)

            rho_traj.append(self.network.rho.data.detach().cpu().numpy())

            if sch:
                sch.step()
            ##### ID Validation
            self.network.eval()
            with torch.no_grad():
                valid_loss_id, valid_pred_id, valid_true_id = 0., [], []
                for b, batch in enumerate(self.id_valid_loader):
                    print('id_valid_loader batch: ', b, f'/ {len(self.id_valid_loader)}' )
                    ##### x
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    ##### y
                    y_ = batch['y'].to(self.args.device).to(torch.float32)  # (B)
                    one_hot = np.zeros((self.args.batch_size, self.args.num_classes))
                    # one_hot = np.zeros((args.batch_size, args.num_classes))
                    for i in range(len(one_hot)):
                        one_hot[i,int(y_[i])] = 1
                    y = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                    ##### s
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    col_idx = [np.where(int(s_[idx].item())  == np.array(self.args.train_domains))[0][0] for idx in range(len(s_))]
                    for i in range(len(one_hot)):
                        one_hot[i,col_idx[i]] = 1
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                    
                    # x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                    # x = train_transform(x.to(torch.uint8))
                    # x = x.float().to(torch.float32)
                    # y_ = batch['y'].to(args.device).to(torch.float32)  # (B)  - CUDA
                    # y_ = batch['y'].to(torch.float32)  # (B) - CPU
                    # one_hot = np.zeros((args.batch_size, args.num_classes))
                    # for i in range(len(one_hot)):
                    #     one_hot[i,int(y_[i])] = 1
                    # y = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (Batch_size, # Nomains)
                    # s_ = batch['domain'].to(args.device).to(torch.float32)  # (Batch_size)
                    # one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                    # col_idx = [np.where(int(s_[idx].item())  == np.array(args.train_domains))[0][0] for idx in range(len(s_))]
                    # for i in range(len(one_hot)):
                    #     one_hot[i,col_idx[i]] = 1
                    # s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, K) s.shape s.sum()
                    
                    batch_probits_f = self.network.forward_f(x)
                    batch_probits_g = self.network.forward_g(x)
                    rho = self.network.rho # (B, J+1)
                    assert rho.shape[1] == batch_probits_f.shape[1]+1

                    y_true = y_.clone()                     # y_true.shape == (B,)
                    s_true = s_.unsqueeze(-1).clone()       # s_true.shape == (B, 1) 
                    y_pred = batch_probits_f.clone()        # y_pred.shape == (B, J) in probits
                    s_pred = batch_probits_g.clone()        # s_pred.shape == (B, K)
                    loss, loss_sel, loss_not_sel = multiclass_classification_loss(self,
                                                                                y_pred=y_pred, # in probits
                                                                                y_true=y_true, 
                                                                                s_pred=s_pred, # in probits
                                                                                s_true=s_true, 
                                                                                rho=rho,
                                                                                approximate=True,)
                    
                    valid_loss_id += loss.item() / len(id_valid_loader) # loss=1.2
                    valid_pred_id.append(F.softmax(y_pred, dim=1).detach().cpu().numpy())
                    valid_true_id.append(y.detach().cpu().numpy())
                    # F.softmax(y_pred, dim=1).shape == y.shape
            
            
            ######### ID valid 
            y_true_set = np.concatenate(valid_true_id)
            y_pred_set = np.concatenate(valid_pred_id)
            idx = y_pred_set.argmax(axis=1)
            y_pred_class = np.zeros((len(y_pred_set), self.args.num_classes)).astype(int)
            # y_pred_class = torch.zeros(y_pred_set.shape[0], args.num_classes, dtype=torch.int)
            for i in range(len(y_pred_class)):
                y_pred_class[i, idx[i]] = 1
            ######### loss, acc, f1
            print('valid_loss_id', valid_loss_id)
            acc_id = accuracy_score(y_true_set, y_pred_class).round(3) 
            f1_id = f1_score(y_true_set, y_pred_class, average='macro', zero_division=0).round(3) ###
            ######### save
            id_valid_loss_traj.append(valid_loss_id)
            id_valid_acc_traj.append(acc_id)
            id_valid_f1_traj.append(f1_id)

            ##### OOD Validation
            self.network.eval()
            with torch.no_grad():
                valid_loss_ood, valid_pred_ood, valid_true_ood = 0., [], []
                for b, batch in enumerate(self.ood_valid_loader):
                    print('ood_valid_loader batch: ', b, f'/ {len(self.ood_valid_loader)}' )
                    ##### x batch = batch_ood
                    x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
                    if self.eval_transform is not None:
                        x = self.eval_transform(x.to(torch.uint8))
                    x = x.float().to(torch.float32)
                    ##### y
                    y_ = batch['y'].to(self.args.device).to(torch.float32)  # (B)
                    one_hot = np.zeros((self.args.batch_size, self.args.num_classes))
                    # one_hot = np.zeros((args.batch_size, args.num_classes))
                    for i in range(len(one_hot)):
                        one_hot[i,int(y_[i])] = 1
                    y = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                    ##### s
                    s_ = batch['domain'].to(self.args.device).to(torch.float32)  # (B)
                    one_hot = np.zeros((self.args.batch_size, len(self.args.train_domains)))
                    s = torch.Tensor(one_hot).to(self.args.device).to(torch.float32) # (B, K)
                    
                    batch_probits_f = self.network.forward_f(x)
                    batch_probits_g = self.network.forward_g(x)
                    rho = self.network.rho # (B, J+1)
                    assert rho.shape[1] == batch_probits_f.shape[1]+1
                    '''
                    x = batch['x'].to(args.device).to(torch.float32)  # (B, *)
                    x = train_transform(x.to(torch.uint8))
                    y_ = batch['y'].to(args.device).to(torch.float32)  # (B)  - CUDA
                    y_ = batch['y'].to(torch.float32)  # (B) - CPU
                    one_hot = np.zeros((args.batch_size, args.num_classes))
                    for i in range(len(one_hot)):
                        one_hot[i,int(y_[i])] = 1
                    y = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (Batch_size, # Nomains)
                    s_ = batch['domain'].to(args.device).to(torch.float32)  # (Batch_size)
                    one_hot = np.zeros((args.batch_size, len(args.train_domains)))
                    s = torch.Tensor(one_hot).to(args.device).to(torch.float32) # (B, K) s.shape s.sum()
                    batch_probits_f = network.forward_f(x)
                    batch_probits_g = network.forward_g(x)
                    rho = network.rho # (B, J+1)
                    assert rho.shape[1] == batch_probits_f.shape[1]+1
                    '''

                    y_true = y_.clone()                     # y_true.shape == (B,)
                    s_true = s_.unsqueeze(-1).clone()       # s_true.shape == (B, 1) 
                    y_pred = batch_probits_f.clone()        # y_pred.shape == (B, J) in probits
                    s_pred = batch_probits_g.clone()        # s_pred.shape == (B, K)
                    
                    # loss, loss_sel, loss_not_sel = multiclass_classification_loss(self,
                    #                                                             y_pred=y_pred, # in probits
                    #                                                             y_true=y_true, 
                    #                                                             s_pred=s_pred, # in probits
                    #                                                             s_true=s_true, 
                    #                                                             rho=rho,
                    #                                                             approximate=True,)
                    
                    # valid_loss_ood += loss.item() / len(ood_valid_loader) # loss=1.2
                    valid_pred_ood.append(F.softmax(y_pred, dim=1).detach().cpu().numpy())
                    valid_true_ood.append(y.detach().cpu().numpy())
                    # F.softmax(y_pred, dim=1).shape == y.shape
            
            ######### OOD valid 
            y_true_set = np.concatenate(valid_true_ood)
            y_pred_set = np.concatenate(valid_pred_ood)
            idx = y_pred_set.argmax(axis=1)
            y_pred_class = np.zeros((len(y_pred_set), self.args.num_classes)).astype(int)
            # y_pred_class = torch.zeros(y_pred_set.shape[0], args.num_classes, dtype=torch.int)
            for i in range(len(y_pred_class)):
                y_pred_class[i, idx[i]] = 1
            ######### loss, acc, f1
            print('valid_loss_ood', valid_loss_ood)
            acc_ood = accuracy_score(y_true_set, y_pred_class).round(3) 
            f1_ood = f1_score(y_true_set, y_pred_class, average='macro', zero_division=0).round(3) ###
            ######### save
            # ood_valid_loss_traj.append(valid_loss_ood)
            ood_valid_acc_traj.append(acc_ood)
            ood_valid_f1_traj.append(f1_ood)


            # model_selection: loss
            print(self.args.model_selection)
            if self.args.model_selection == 'loss':
                # valid_loss = (self.args.w * valid_loss_id) + valid_loss_ood
                valid_loss = (self.args.w * valid_loss_id) + ((1-self.args.w) * valid_loss_ood)
                if valid_loss < best_loss:
                    best_model = deepcopy(self.network.state_dict())
                    best_loss = valid_loss
                    best_epoch = epoch
                else:
                    pass
            # model_selection: metric
            elif self.args.model_selection == 'metric':
                # if self.args.model_selection_metric=='auc':
                #     # valid_metric = (self.args.w * auc_id) + auc_ood
                #     valid_metric = (self.args.w * auc_id) + ((1-self.args.w) * auc_ood)
                if self.args.model_selection_metric=='f1':
                    # valid_metric = (self.args.w * f1_id) + f1_ood
                    valid_metric = (self.args.w * f1_id) + ((1-self.args.w) * f1_ood)
                elif self.args.model_selection_metric=='accuracy':
                    # valid_metric = (self.args.w * acc_id) + acc_ood)
                    valid_metric = (self.args.w * acc_id) + ((1-self.args.w) * acc_ood)
                else:
                    print('choose f1, or accuracy')
                if valid_metric > best_metric:
                    best_model = deepcopy(self.network.state_dict())
                    best_metric = valid_metric
                    best_epoch = epoch
                else:
                    pass
            else:
                print('choose model selection type')            
            
            print('epoch done')
            desc = f'[{epoch + 1:03d}/{self.args.epochs}] '
            # desc = f'[{epoch + 1:03d}/{args.epochs}] '
            # Training
            desc += f'| train | Loss: {train_loss_traj[-1]:.4f} '
            # desc += f'tr_auc: {train_auc_traj[-1]:.4f} '
            desc += f'tr_acc: {train_acc_traj[-1]:.4f} '
            desc += f'tr_f1: {train_f1_traj[-1]:.4f} '
            # ID validation
            desc += f'| ID valid | Loss: {id_valid_loss_traj[-1]:.4f} '
            # desc += f'id_val_auc: {id_valid_auc_traj[-1]:.4f} '
            desc += f'id_val_acc: {id_valid_acc_traj[-1]:.4f} '
            desc += f'id_val_f1: {id_valid_f1_traj[-1]:.4f} '
            # OOD validation
            # desc += f'| OOD valid | Loss: {ood_valid_loss_traj[-1]:.4f} '
            # desc += f'ood_val_auc: {ood_valid_auc_traj[-1]:.4f} '
            desc += f'ood_val_acc: {ood_valid_acc_traj[-1]:.4f} '
            desc += f'ood_val_f1: {ood_valid_f1_traj[-1]:.4f} '
            pbar.set_description(desc)
        # selected model
        self.network.load_state_dict(best_model)
        
        # Traj Loss & Metric: Training
        self.train_loss_traj = train_loss_traj
        self.train_auc_traj = train_auc_traj
        self.train_f1_traj = train_f1_traj
        # Traj Loss & Metric: ID validation
        self.id_valid_loss_traj = id_valid_loss_traj
        self.id_valid_acc_traj = id_valid_acc_traj
        self.id_valid_f1_traj = id_valid_f1_traj
        # Traj Loss & Metric: OOD validation
        # self.ood_valid_loss_traj = ood_valid_loss_traj
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
        self.best_train_acc = self.train_acc_traj[best_epoch]
        self.best_train_f1 = self.train_f1_traj[best_epoch]
        # Best Loss & Metric: Training
        self.best_id_valid_loss = self.id_valid_loss_traj[best_epoch]
        self.best_id_valid_acc = self.id_valid_acc_traj[best_epoch]
        self.best_id_valid_f1 = self.id_valid_f1_traj[best_epoch]
        # Best Loss & Metric: OOD Validation
        # self.best_ood_valid_loss = self.ood_valid_loss_traj[best_epoch]
        self.best_ood_valid_acc = self.ood_valid_acc_traj[best_epoch]
        self.best_ood_valid_f1 = self.ood_valid_f1_traj[best_epoch]
        # Best Loss & best_epoch
        self.best_epoch = best_epoch
        if self.args.model_selection == 'loss':
            self.best_valid_metric = best_loss
        elif self.args.model_selection == 'metric':
            self.best_valid_metric = best_metric

    def predict_proba(self, batch):
        x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
        with torch.no_grad():
            probs_batch = F.softmax(self.network.forward_f(x.to(self.args.device)), dim=1)
        probs_batch = probs_batch.detach().cpu().numpy()
        return probs_batch
    
    def predict_proba_loader(self, dataloader):
        probs_list=[]
        for b, batch in enumerate(dataloader):
            print('predict batch', b, f'/ {len(dataloader)}')
            x = batch['x'].to(self.args.device).to(torch.float32)  # (B, *)
            with torch.no_grad():
                probs_batch = F.softmax(self.network.forward_f(x.to(self.args.device)), dim=1)
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
