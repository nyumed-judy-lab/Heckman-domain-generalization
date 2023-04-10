
import os
import abc
import typing

import torch
import torch.nn as nn

from torch.utils.data import Dataset


class MultipleDomainCollection(object):
    def __init__(self):
        
        self.train_domains = list()
        self.validation_domains = list()
        self.test_domains = list()
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._test_datasets = list()

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_train_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                    typing.List[Dataset]]:
        if len(self._train_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.train_domains, self._train_datasets)}
            return self._train_datasets

    def get_id_validation_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                            typing.List[Dataset]]:
        if len(self._id_validation_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.train_domains, self._id_validation_datasets)}
            return self._id_validation_datasets

    def get_ood_validation_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                             typing.List[Dataset]]:
        if len(self._ood_validation_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.validation_domains, self._ood_validation_datasets)}
            return self._ood_validation_datasets

    def get_test_data(self, as_dict: bool = False) -> typing.Union[typing.Dict[str, Dataset],
                                                                   typing.List[Dataset]]:
        if len(self._test_datasets) == 0:
            return None
        else:
            if as_dict:
                return {n: d for n, d in zip(self.test_domains, self._test_datasets)}
            return self._test_datasets

    @property
    def input_shape(self) -> tuple:
        raise NotImplementedError
    
    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    @staticmethod
    def as_list(x: typing.Union[int, typing.Iterable[int]]) -> typing.List[int]:
        if x is None:
            return []
        elif isinstance(x, int):
            return [x]
        elif isinstance(x, list):
            return x
        elif isinstance(x, tuple):
            return list(x)
        elif isinstance(x, dict):
            return [v for _, v in x.items()]
        else:
            raise NotImplementedError


MultipleDomainDataset = MultipleDomainCollection


class MultiEnvDataset(object):
    def __getitem__(self, index: int):
        return self.datasets[index]
    
    def __len__(self):
        return len(self.datasets)

    @property
    def train_datasets(self):
        raise NotImplementedError("Override within subclass.")
    
    @property
    def test_dataset(self):
        raise NotImplementedError("Override within subclass.")

    @property
    def num_environments(self):
        raise NotImplementedError("Override within subclass.")

    @property
    def input_shape(self):
        raise NotImplementedError("Override within subclass.")
    
    @property
    def num_classes(self):
        raise NotImplementedError("Override within subclass.")
