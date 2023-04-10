
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_cross_entropy_with_probits(probits: torch.FloatTensor,
                                      targets: typing.Union[torch.LongTensor, torch.FloatTensor],
                                      reduction: str = 'mean',
                                      **kwargs) -> torch.FloatTensor:
    """
    Binary cross entropy with probits.
    Arguments:
        probits: `torch.FloatTensor` of shape (B, C) or (B,  )
        targets: `torch.FloatTensor` of shape (B, C) or (B,  )
    """
    if (probits.ndim == 2) and (targets.ndim == 2):
        if (probits.size(1) == 1) and (targets.size(1) == 1):
            probits, targets = probits.squeeze(), targets.squeeze()
    probs = torch.distributions.Normal(loc=0., scale=1.).cdf(probits)
    return F.binary_cross_entropy(probs, targets.float(), reduction=reduction, **kwargs)



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
        a: 1D `torch.FloatTensor` of shape (B, )
        b: 1D `torch.FloatTensor` of shape (B, )
        rho:
    """
    device = a.device
    a, b = a.view(-1, 1), b.view(-1, 1)  # for proper broadcasting with x
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    x = _linspace_with_grads(start=0, stop=rho, steps=steps,
                             device=device).unsqueeze(0).expand(a.shape[0], -1)  # (B, steps)
    y = 1 / torch.sqrt(1 - torch.pow(x, 2)) * torch.exp(
        - (torch.pow(a, 2) + torch.pow(b, 2) - 2 * a * b * x) / (2 * (1 - torch.pow(x, 2)))
    )

    return _normal.cdf(a.squeeze()) * _normal.cdf(b.squeeze()) + \
        (1 / (2 * torch.pi)) * torch.trapezoid(y=y, x=x)


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


def _cross_domain_regression_loss(self,
                                    y_pred: torch.FloatTensor,
                                    y_true: torch.FloatTensor,
                                    s_pred: torch.FloatTensor,
                                    s_true: torch.LongTensor,
                                    rho   : torch.FloatTensor,
                                    sigma : torch.FloatTensor, ) -> torch.FloatTensor:
    """
    Arguments:
        y_pred : 1d `torch.FloatTensor` of shape (N,  ),
        y_true : 1d `torch.FloatTensor` of shape (N,  ),
        s_pred : 2d `torch.FloatTensor` of shape (N, K),
        s_true : 1d `torch.LongTensor`  of shape (N,  ),
        rho    : 1d `torch.FloatTensor` of shape (N,  ),
        sigma  : 1d `torch.FloatTensor` of shape (1,  ), singleton.
    """

    if (y_pred.ndim == 2) and (y_pred.size(1) == 1):
        y_pred = y_pred.squeeze(1)
    if (y_true.ndim == 2) and (y_true.size(1) == 1):
        y_true = y_true.squeeze(1)

    _epsilon: float = 1e-7
    _normal = torch.distributions.Normal(loc=0., scale=1.)

    # Gather values from `s_pred` those that correspond to the true domains
    s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).squeeze()

    # -\log p[S_k = 1, y] = -\log p(y) -\log p(S_k = 1 | y) ; shape = (N,  )
    loss_selected = - torch.log(
        _normal.cdf(
            (s_pred_k + rho * (y_true - y_pred).div(_epsilon + sigma)) / (_epsilon + torch.sqrt(1 - rho ** 2))
        ) + _epsilon
    ) + 0.5 * (
        torch.log(2 * torch.pi * (sigma ** 2)) \
            + F.mse_loss(y_pred, y_true, reduction='none').div(_epsilon + sigma ** 2)
    )

    if (self.args.freeze_selection_encoder & self.args.freeze_selection_head):
        return loss_selected.mean()

    # domain indicators in 2d; (N, K) <- (N,  )
    s_true_2d = torch.zeros_like(s_pred).scatter_(
        dim=1,
        index=s_true.view(-1, 1),
        src=torch.ones_like(s_pred),
    )

    # -\log Pr[S_l = 0] for l \neq k
    #   shape; (N(K-1),  )
    loss_not_selected = - torch.log(1 - _normal.cdf(s_pred) + _epsilon)                    # (N     , K)
    loss_not_selected = torch.nan_to_num(loss_not_selected, nan=0., posinf=0., neginf=0.)  # (N     , K)
    loss_not_selected = loss_not_selected.masked_select((1 - s_true_2d).bool())            # (N(K-1),  )

    return torch.cat([loss_selected, loss_not_selected], dim=0).mean(), loss_selected.mean(), loss_not_selected.mean()

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

def _cross_domain_multiclass_classification_loss(self,
                                                    y_pred: torch.FloatTensor,
                                                    y_true: torch.LongTensor,
                                                    s_pred: torch.FloatTensor,
                                                    s_true: torch.LongTensor,
                                                    rho: torch.FloatTensor,       # shape; (N, J+1)
                                                    approximate: bool = False,    # logistic approx.
                                                    **kwargs, ) -> torch.FloatTensor:
    """Multinomial loss with logistic approximation available."""

    _eps: float = 1e-7
    _normal = torch.distributions.Normal(loc=0., scale=1.)
    _float_type = y_pred.dtype  # creating new tensors (in case of amp)

    B = int(y_true.size(0))  # batch size
    J = int(y_pred.size(1))  # number of classes
    K = int(s_pred.size(1))  # number of (training) domains

    s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).flatten()
    assert len(s_pred_k) == len(s_pred)

    # matrix of y(probit) differences (with respect to its true outcome)
    y_pred_diff = torch.zeros(B, J-1, dtype=_float_type, device=self.device)
    for j in range(J):
        col_mask = torch.arange(0, J, device=self.device).not_equal(j)
        row_mask = y_true.eq(j)
        y_pred_diff[row_mask, :] = (
            y_pred[:, j].view(-1, 1) - y_pred[:, col_mask]
        )[row_mask, :]

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