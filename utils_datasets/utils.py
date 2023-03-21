
import random
import bisect

import torch
import torch.nn as nn
import numpy as np

from typing import Iterable, Tuple, Union, List

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold


class DictionaryDataset(Dataset):
    """Wrapper for data in dictionaries."""
    def __init__(self, data: dict, exclude_keys: List[str] = []):
        self.data = data
        self.exclude_keys = exclude_keys
        self.keys = [c for c in self.data if c not in exclude_keys]

    def __getitem__(self, index: int) -> dict:
        out = dict()
        for k in self.data:
            if k in self.exclude_keys:
                continue
            out[k] = self.data[k][index]
        
        return out

    def __len__(self):
        return self.data[self.keys[0]].__len__()


def get_domain_balanced_sampler(domain_lengths: List[int]) -> WeightedRandomSampler:
    """Returns a weighted sampler inversely proportional to domain lengths."""
    domain_counts = torch.LongTensor(domain_lengths)
    domain_weights = 1 / domain_counts
    domain_indicators = list()
    for i, l in enumerate(domain_lengths):
        domain_indicators += [i] * l
    domain_indicators = torch.LongTensor(domain_indicators)
    sample_weights = domain_weights[domain_indicators]
    
    return WeightedRandomSampler(weights=sample_weights,
                                 num_samples=sum(domain_lengths),
                                 replacement=True)


class CVFoldBase(object):
    def __init__(self, dataset: Union[torch.utils.data.Dataset, Iterable]):
        if isinstance(dataset, torch.utils.data.Dataset):
            self.dataset = dataset
        elif isinstance(dataset, Iterable):
            assert all([isinstance(d, torch.utils.data.Dataset) for d in dataset])
            self.dataset = ConcatDataset(dataset)
        else:
            raise NotImplementedError


class PlainCVFold(CVFoldBase):
    def __init__(self, dataset: Union[torch.utils.data.Dataset, Iterable]):
        super(PlainCVFold, self).__init__(dataset=dataset)

    def split(self, n_splits: int, random_state: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:
        size = len(self.dataset)
        _indices = np.arange(size)
        _splitter = KFold(n_splits, shuffle=True, random_state=random_state)
        for train_indices, validation_indices in _splitter.split(_indices):
            train_indices = torch.from_numpy(train_indices)
            validation_indices = torch.from_numpy(validation_indices)
            yield train_indices, validation_indices


class DomainStratifiedCVFold(CVFoldBase):
    def __init__(self, dataset: Union[torch.utils.data.Dataset, Iterable]):
        super(DomainStratifiedCVFold, self).__init__(dataset=dataset)
    
    def split(self, n_splits: int, domain: torch.LongTensor, random_state: int = 0) -> Tuple[torch.LongTensor, torch.LongTensor]:
        size = len(self.dataset)
        _indices = np.arange(size)
        _domain = domain.cpu().numpy()
        _splitter = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        for train_indices, validation_indices in _splitter.split(X=_indices, y=_domain):
            train_indices = torch.from_numpy(train_indices)
            validation_indices = torch.from_numpy(validation_indices)
            yield train_indices, validation_indices


class LeaveOneDomainOut(CVFoldBase):
    def __init__(self, dataset: Union[torch.utils.data.Dataset, Iterable]):
        super(LeaveOneDomainOut, self).__init__(dataset=dataset)

    def split(self, domain: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
        size = len(self.dataset)
        _indices = np.arange(size)
        _domain = domain.cpu().numpy()
        for train_indices, validation_indices in LeaveOneGroupOut().split(X=_indices, groups=_domain):
            train_indices = torch.from_numpy(train_indices)
            validation_indices = torch.from_numpy(validation_indices)
            yield train_indices, validation_indices


class TransformWrapper(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset,
                       transform: nn.Sequential = None) -> None:
        """A wrapper for input data transforms."""
        super(TransformWrapper, self).__init__()
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Iterable[torch.Tensor]:
        x, y, e = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y, e


class FastMultiEnvTriplet(Dataset):
    def __init__(self, datasets: Union[Dataset, Iterable[Dataset]],
                       transform: nn.Sequential = None,
                       device: str = 'cpu') -> None:
        super(FastMultiEnvTriplet, self).__init__()
        
        if isinstance(datasets, (list, tuple)):
            for dataset in datasets:
                if len(dataset[0]) != 3:
                    raise NotImplementedError
            self.dataset = ConcatDataset(datasets)
        else:
            self.dataset = datasets  # use as-is
        self.transform = transform

        with torch.no_grad():
            y = torch.stack([y for _, y, _ in self.dataset], dim=0)
            e = torch.stack([e for _, _, e in self.dataset], dim=0)
            y_cond: torch.BoolTensor = y.view(-1, 1).eq(y.view(1, -1))
            e_cond: torch.BoolTensor = e.view(-1, 1).eq(e.view(1, -1))
            self.lpos_mapper = torch.multinomial((y_cond & (~e_cond)).to(device).float(), num_samples=1, replacement=False)
            self.lpos_mapper = self.lpos_mapper.squeeze()
            self.lpos_mapper = self.lpos_mapper.to('cpu')
            self.dpos_mapper = torch.multinomial(((~y_cond) & e_cond).to(device).float(), num_samples=1, replacement=False)
            self.dpos_mapper = self.dpos_mapper.squeeze()
            self.dpos_mapper = self.dpos_mapper.to('cpu')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Iterable[Tuple[torch.Tensor, ...]]:
        xi, yi, ei = self.dataset[idx]
        xj, yj, ej = self.dataset[self.lpos_mapper[idx]]
        xk, yk, ek = self.dataset[self.dpos_mapper[idx]]
        if self.transform is not None:
            xi, xj, xk = list(map(self.transform, [xi, xj, xk]))
        return (xi, yi, ei), (xj, yj, ej), (xk, yk, ek)


class MultiEnvTriplet(Dataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(MultiEnvTriplet, self).__init__()
        self.datasets = list(datasets)
        for dataset in self.datasets:
            if len(dataset[0]) != 3:
                raise NotImplementedError
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Iterable[tuple]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length.")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        # 1. anchor sample
        xi, yi, ei = self.datasets[dataset_idx][sample_idx]
        # 2. label-positive, domain-negative sample
        lpos_dataset_idx: int = random.sample([e for e in range(len(self.datasets)) if e != dataset_idx], k=1)[0]
        lpos_dataset: TensorDataset = self.datasets[lpos_dataset_idx]
        lpos_candidates = [(x, y, e) for (x, y, e) in lpos_dataset if y == yi]
        xj, yj, ej = random.sample(lpos_candidates, k=1)[0]
        # 3. label-negative, domain-positive sample
        dpos_dataset_idx: int = dataset_idx
        dpos_dataset: TensorDataset = self.datasets[dpos_dataset_idx]
        dpos_candidates = [(x, y, e) for (x, y, e) in dpos_dataset if y != yi]
        xk, yk, ek = random.sample(dpos_candidates, k=1)[0]

        return (xi, yi, ei), (xj, yj, ej), (xk, yk, ek)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r
