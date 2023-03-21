
from typing import Iterator, Sequence
from torch.utils.data import Sampler, Dataset
import torch


def inverse_class_frequency_sample_weights(targets: torch.LongTensor) -> torch.Tensor:
    """
    Create sample weights based on inverse class frequencies.
    Arguments:
        targets: 1D LongTensor of shape (N, )
    Returns:
        1D FloatTensor of shape (N, )
    """
    unique_classes, class_counts = torch.unique(targets, sorted=True, return_counts=True)
    unique_classes: list = unique_classes.cpu().tolist()
    class_counts: list = class_counts.cpu().tolist()
    class_weights: dict = {
        u: len(targets) / class_counts[i] for i, u in enumerate(unique_classes)
    }
    sample_weights = [class_weights[y] for y in targets.cpu().tolist()]
    return torch.tensor(sample_weights)


class GroupSampler(Sampler):
    """
    Constructs batches by first sampling groups,
    then sampling data from those groups.
    Drops the last batch if it is incomplete.
    Note that this is a batch sampler.
    """
    def __init__(self,
                 group_ids: torch.Tensor,
                 batch_size: int,
                 n_groups_per_batch: int = 1,
                 uniform_over_groups: bool = True,
                 distinct_groups: bool = True):
        
        # Check argument compatibility
        if batch_size % n_groups_per_batch != 0:
            raise ValueError
        if len(group_ids) < batch_size:
            raise ValueError

        self.group_ids: torch.LongTensor = group_ids
        self.unique_groups, self.group_indices, self.unique_counts = self.split_into_groups(group_ids)

        self.batch_size: int = batch_size
        self.n_groups_per_batch: int = n_groups_per_batch
        self.n_points_per_group: int = self.batch_size // self.n_groups_per_batch

        self.uniform_over_groups: bool = uniform_over_groups
        if not distinct_groups:
            raise NotImplementedError
        self.distinct_groups: bool = distinct_groups

        self.dataset_size: int = len(group_ids)
        self.num_batches: int = self.dataset_size // batch_size

        if self.uniform_over_groups:
            self.group_prob: torch.FloatTensor = torch.ones(len(self.unique_counts), dtype=torch.float)
        else:
            self.group_prob: torch.FloatTensor = self.unique_counts.div(self.dataset_size)
    
    def __iter__(self):

        for _ in range(self.num_batches):
            
            # select group indices according to `self.n_groups_per_batch`
            groups_for_batch = torch.multinomial(self.group_prob, 
                                                 num_samples=self.n_groups_per_batch,
                                                 replacement=not self.distinct_groups,)
            
            # sample points for each group
            sampled_ids = list()
            for group in groups_for_batch:
                candidates = self.group_indices[group]
                sample_idx = torch.multinomial(torch.ones_like(candidates).float(),
                                               num_samples=self.n_points_per_group,
                                               replacement=len(candidates) <= self.n_points_per_group,)
                sampled_ids += [candidates[sample_idx]]
            
            # flatten
            sampled_ids = torch.cat(sampled_ids, dim=0)
            
            yield sampled_ids

    def __len__(self):
        return self.num_batches
    
    @staticmethod
    def split_into_groups(g: torch.Tensor, sort: bool = True):
        """
        Arguments:
            g: 1d tensor of group indicators
        Returns:
            unique_groups: 1d tensor of unique groups
            group_indices: list of 1d tensors where the i-th tensor is the
                indices of elements of g that equal unique_groups[i].
            unique_counts: 1d tensor of counts of each element in unique_groups.
        """
        unique_groups, unique_counts = torch.unique(g, sorted=sort, return_counts=True)
        group_indices = list()
        for group in unique_groups:
            group_indices += [torch.nonzero(g == group, as_tuple=True)[0]]
        
        return unique_groups, group_indices, unique_counts


class DatasetBalancedSampler(Sampler):
    def __init__(self,
                 datasets: Sequence[Dataset],
                 num_samples: int = None,
                 replacement: bool = True,
                 generator=None,
                 ) -> None:
        
        lengths = list(map(len, datasets))
        lengths = torch.LongTensor(lengths)

        indicators = list()
        for i, l in enumerate(lengths):
            indicators += [i] * l
        indicators = torch.LongTensor(indicators)
        self.weights = lengths.reciprocal()[indicators]
        
        if num_samples is not None:
            if not isinstance(num_samples, int):
                raise ValueError
            self.num_samples = num_samples
        else:
            self.num_samples = len(self.weights)
        
        self.replacement = replacement
        self.generator = generator
        
    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights, self.num_samples, self.replacement, generator=self.generator
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples


class GroupBalancedSampler(Sampler):
    def __init__(self,
                 groups: torch.LongTensor,
                 num_samples: int = None,
                 replacement: bool = True,
                 generator=None,
                 ) -> None:
        """
        Arguments:
            groups: {1D, 2D} LongTensor; unique_groups are calculated across dim=0.
        """
        self.unique_groups, self.group_counts = groups.unique(return_counts=True, dim=0)
        self.group_weights = self.group_counts.reciprocal()
        if self.unique_groups.ndim == 1:
            self.sample_weights = self.group_weights[groups]
        elif self.unique_groups.ndim == 2:
            self.sample_weights = torch.zeros(len(groups), dtype=torch.float32)
            for i, g in enumerate(self.unique_groups):
                mask = groups.eq(g).sum(dim=1).eq(groups.size(1)).view(-1)
                self.sample_weights[mask] = self.group_weights[i]
        else:
            raise NotImplementedError

        if num_samples is not None:
            assert isinstance(num_samples, int)
            self.num_samples: int = num_samples
        else:
            self.num_samples: int = len(self.sample_weights)
        
        self.replacement: bool = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.sample_weights, num_samples=self.num_samples,
            replacement=self.replacement, generator=self.generator,
        )
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        _repr = list()
        i: int = 0
        for _, count, weight in zip(self.unique_groups, self.group_counts, self.group_weights):
            _repr.append(f"group={i}, count={count:,}, weight={weight * 100:.4f}")
            i += 1
        return "| ".join(_repr)


def configure_train_sampler(datasets: Sequence[Dataset],
                            use_domain: bool = True, 
                            use_target: bool = False) -> torch.utils.data.Sampler:
    
    lengths: list = [len(dataset) for dataset in datasets]
    n: int = sum(lengths)
    domains, targets = list(), list()
    
    if (not use_domain) and (not use_target):
        return None

    if use_domain:
        for dataset in datasets:
            domains += [
                torch.tensor(
                    [dataset.get_domain(i) for i in range(len(dataset))]
                )
            ]
        domains = torch.cat(domains, dim=0)
    else:
        domains = torch.zeros(n, dtype=torch.long)
    
    if use_target:
        for dataset in datasets:
            targets += [
                torch.tensor(
                    [dataset.get_target(i) for i in range(len(dataset))]
                )
            ]
        targets = torch.cat(targets, dim=0)
        if targets.dtype != torch.long:
            raise ValueError(f"targets have unsupported dtype: {targets.dtype}")
    else:
        targets = torch.zeros(n, dtype=torch.long)

    groups = torch.stack([targets, domains], dim=1)  # (N, 2)
    return GroupBalancedSampler(groups)
