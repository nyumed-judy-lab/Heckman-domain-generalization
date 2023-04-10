import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _linspace_with_grads(start: torch.Tensor, stop: torch.Tensor, steps: int, device: str = 'cpu'):
    """
    Creates a 1-dimensional grid while preserving the gradients.
    Reference:
        https://github.com/esa/torchquad/blob/4be241e8462949abcc8f1ace48d7f8e5ee7dc136/torchquad/integration/utils.py#L7
    """
    grid = torch.linspace(0, 1, steps, device=device)  # Create a 0 to 1 spaced grid
    grid *= stop - start                               # Scale to desired range, keeping gradients
    grid += start                                      # TODO; sanity check
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
    a, b, rho = batch_probits_f, batch_probits_g[:, k], network.rho[k]
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
'''
a: batch_probits_f
b: batch_probits_g[:, k]torch.round(, decimals=2) batch_probits_g.round(3) output of selection model
c: rho
'''

safe_log = lambda x: torch.log(torch.clip(x, 1e-6))

'''
Equation 20 in ICLR paper
'''
loss = 0.
# for k in range(self.args.num_domains):
for k in range(args.num_domains):
    # k=0
    # joint_prob = bivariate_normal_cdf(batch_probits[:, 0], batch_probits_g[:, k], self.network.rho[k])
    joint_prob = bivariate_normal_cdf(batch_probits_f, batch_probits_g[:, k], network.rho[k])
    loss += -(y * s[:, k] * safe_log(joint_prob)).mean() 
    loss += -((1 - y) * s[:, k] * safe_log(normal.cdf(batch_probits_g[:, k]) - joint_prob)).mean()
    loss += -((1 - s[:, k]) * safe_log(normal.cdf(-batch_probits_g[:, k]))).mean()



def heckman_classification_nll(y_pred: torch.FloatTensor, y_true: torch.LongTensor,
                               s_pred: torch.FloatTensor, s_true: torch.LongTensor,
                               rho: typing.Union[float, torch.FloatTensor],
                               ) -> torch.FloatTensor:
    """
    Heckman correction loss for binary classification.
    Arguments:
        y_pred:
        y_true:
        s_pred:
        s_true:
        rho:
    Returns:
    """

    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()  # in probits
    s_true = s_true.squeeze()
    s_pred = s_pred.squeeze()  # in probits

    _epsilon = 1e-5
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    # Loss for unselected individuals
    loss_not_selected = - (1 - s_true) * torch.log(1 - _normal.cdf(s_pred) + _epsilon)
    loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0.)

    # Loss for selected individuals with positive outcomes
    loss_selected_pos = - s_true * y_true * torch.log(
        bivariate_normal_cdf(a=s_pred, b=y_pred, rho=rho) + _epsilon
    )
    loss_selected_pos = torch.nan_to_num(loss_selected_pos, nan=0.)

    # Loss for selected individuals with negative outcomes
    """
    loss_selected_neg = - s_true * (1 - y_true) * torch.log(
        bivariate_normal_cdf(a=s_pred, b=-y_pred, rho=-rho) + _epsilon
    )
    """

    loss_selected_neg = - s_true * (1 - y_true) * torch.log(
        _normal.cdf(s_pred) - bivariate_normal_cdf(a=s_pred, b=y_pred, rho=rho) + _epsilon
    )
    loss_selected_neg = torch.nan_to_num(loss_selected_neg, nan=0.)

    return (loss_not_selected + loss_selected_pos + loss_selected_neg).mean()

def _cross_domain_binary_classification_loss(self,
                                                y_pred: torch.FloatTensor,
                                                y_true: torch.LongTensor,
                                                s_pred: torch.FloatTensor,
                                                s_true: torch.LongTensor,
                                                rho   : torch.FloatTensor, ) -> torch.FloatTensor:
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
