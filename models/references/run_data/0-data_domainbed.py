# /gpfs/home/choy07/.conda/envs/torch/bin/python

import typing
import os
import time
import glob
import typing
import pathlib
import gdown
import tarfile
import numpy as np
import torch
import matplotlib.pyplot as plt

plt.cm.tab10
K=3
torch.Size([1206, 2])   
s_pred_in_probits = torch.Tensor([
              [0.1, 0.9],
              [0.1, 0.9],
              [0.1, 0.9],
              [0.1, 0.9],
              [0.1, 0.9],
              [0.9, 0.1],
              [0.9, 0.1],
              [0.9, 0.1],
              [0.9, 0.1],
              [0.9, 0.1],])
N, K = s_pred_in_probits.shape
import pandas as pd
import seaborn as sns
# create dataframe
df = pd.DataFrame(s_pred_in_probits.detach().cpu().numpy())
df.columns = [f"probits_{k}" for k in range(K)]
df['domain'] = torch.Tensor([0,0,0,0,0,1,1,1,1,1])


fig, axes = plt.subplots(K, 1, figsize=(12, 8))
# plot histograms
for k in range(K):
    sns.histplot(data=df,
                    x=f'probits_{k}', hue='domain', palette=plt.cm.tab10,
                    bins=1000, stat='count', ax=axes[k])
for k in range(K):
    sns.histplot(data=df,
                    x=f'probits_{k}', hue='domain', palette='tab10',
                    bins=1000, stat='count', ax=axes[k])

sns.histplot(data=df,
                x=f'probits_{k}', hue='domain', palette=plt.cm.tab10,
                bins=1000, stat='count', ax=axes[k])

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from utils_datasets.base import MultipleDomainCollection
from utils_datasets.utils import DictionaryDataset
from utils_datasets.domainbed_ import PACS, VLCS, DomainVLCS

# dataset = VLCS(root = 'data/domainbed/vlcs/')
# dataset.train_domains
# dataset.test_domains
# dataset.validation_domains

# root = 'data/domainbed/vlcs/'
# input_files = np.array(glob.glob(os.path.join(root, "**/*.jpg"), recursive=True))
# env_strings = np.array([pathlib.Path(f).parent.parent.name for f in input_files])

from utils_datasets.domainbed_ import PACS, VLCS, DomainVLCS, SingleVLCS
# VLCS(root = 'data/domainbed/vlcs/',)._download()
dataset = DomainVLCS(root = 'data/domainbed/vlcs/')
dataset.train_domains
dataset.train_domains
dataset.train_domains

train_sets: typing.List[torch.utils.data.Dataset] = dataset.get_train_data(as_dict=False)
id_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_id_validation_data(as_dict=False)
ood_validation_sets: typing.List[torch.utils.data.Dataset] = dataset.get_ood_validation_data(as_dict=False)  # or `None`


from utils_datasets.samplers import configure_train_sampler
from torch.utils.data import ConcatDataset, DataLoader

# Configure train sampler
train_sampler = configure_train_sampler(
    datasets= train_sets,
    use_domain= False,
    use_target= False,
)

_fit_kwargs: typing.Dict[str, object] = {
    'train_set': ConcatDataset(train_sets),
    'id_validation_set': ConcatDataset(id_validation_sets),
    'ood_validation_set': ConcatDataset(ood_validation_sets),
    'train_sampler': train_sampler,
}

##### Example 3


fp_tensor4 = torch.tensor(['a','b'])
fp_tensor4 = torch.tensor([1,2,3])
list1 = fp_tensor4.tolist()
print(list1)
type(list1)
#####
# return self._fit(**_fit_kwargs)  # TODO: add support for test set

train_loader_configs = dict(
    batch_size=512,
    shuffle=train_sampler is None,
    sampler=train_sampler,
    drop_last=True,
    pin_memory=True,
    prefetch_factor=self.args.prefetch_factor,
    num_workers=self.args.num_workers,
)
train_loader = DataLoader(train_set, **train_loader_configs)

domain = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,                                                                                                                                                                                                                                                                                                                                                                                                                                        
        1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1]

domain = torch.tensor(domain)
dataset.test_domains
dataset.train_domains
dataset.validation_domains

s_true_2d = domain.view(-1, 1).eq(torch.tensor(dataset.train_domains, dtype=torch.long).view(1, -1)).long()     
s_true_1d = s_true_2d.nonzero(as_tuple=True)[1]  # (B,  )
assert len(s_true_2d) == len(s_true_1d), "Only supports data with non-overlapping domains."

torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))

# setattr(defaults, 'train_domains', dataset.train_domains)
# setattr(defaults, 'validation_domains', dataset.validation_domains)
# setattr(defaults, 'test_domains', dataset.test_domains)

# PACS(root = 'data/domainbed/pacs/',)._download()
# dataset = PACS(
#         root=defaults.root,
#         train_environments=defaults.train_environments,
#         test_environments=defaults.test_environments,
#         holdout_fraction=defaults.holdout_fraction,
#         download=False,
#     )


#################### matrix operation
import torch
A = torch.randn(4, 4)
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

y_true = torch.randint(0,2, (5,5))
y_pred = torch.randint(0,2, (5,5))
s_pred = torch.randint(0,2, (5,2))
s_true = torch.randint(0,2, (5,2))
B = int(y_true.size(0))  # batch size
J = int(y_pred.size(1))  # number of classes
K = int(s_pred.size(1))  # number of (training) domains
# rho: torch.FloatTensor,       # shape; (N, J+1)
rho = torch.randn(5,J+1).int()      # shape; (N, J+1)
s_pred_k = s_pred.gather(dim=1, index=s_true.view(-1, 1)).flatten()
assert len(s_pred_k) == len(s_pred)

L = torch.eye(J + 1, device=rho.device)  # construct a lower triangular matrix
L[-1] = rho[0]                           # fill in params

L = torch.Tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000, -0.7281],                                                                                                                                
        [ 0.0000,  1.0000,  0.0000,  0.0000,  0.0000, -0.4014],                                                                                                                                    
        [ 0.0000,  0.0000,  1.0000,  0.0000,  0.0000, -0.9899],                                                                                                                                    
        [ 0.0000,  0.0000,  0.0000,  1.0000,  0.0000,  0.1442],                                                                                                                                    
        [ 0.0000,  0.0000,  0.0000,  0.0000,  1.0000, -0.1203],                                                                                                                                    
        [-0.7281, -0.4014, -0.9899,  0.1442, -0.1203,  1.9795]])
D = torch.Tensor([[1.0000e+00, 3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04],                                                                                                              
        [3.1623e-04, 1.0000e+00, 3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04],                                                                                                                  
        [3.1623e-04, 3.1623e-04, 1.0000e+00, 3.1623e-04, 3.1623e-04, 3.1623e-04],                                                                                                                  
        [3.1623e-04, 3.1623e-04, 3.1623e-04, 1.0000e+00, 3.1623e-04, 3.1623e-04],
        [3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04, 1.0000e+00, 3.1623e-04],
        [3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04, 3.1623e-04, 1.4069e+00]])
# C = MatrixOps.cov_to_corr(L @ L.T)       # (J+1, J+1)
D = torch.sqrt(L.diag().diag() + 1e-7)
DInv = torch.linalg.inv(D)
(DInv @ L @ DInv).shape

torch.__version__

A = torch.randn(2, 3, 4, 4)  # Batch of matrices
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

A = torch.randn(4, 4, dtype=torch.complex128)  # Complex matrix
Ainv = torch.linalg.inv(A)
torch.dist(A @ Ainv, torch.eye(4))

L @ L.T