
import os
import time
import typing
import functools

import torch
import numpy as np
import pandas as pd

from ray.util.multiprocessing import Pool as RayPool
from torchvision.io import read_image, ImageReadMode
from utils_datasets.base import MultipleDomainCollection

def subsample_idxs(idxs: np.ndarray, num: int = 5000, take_rest: bool = False, seed= None) -> np.ndarray:
    """
    Reference:
        https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/wilds/common/utils.py#L104
    """
    seed = (seed + 541433) if seed is not None else None
    rng = np.random.default_rng(seed)

    idxs = idxs.copy()
    rng.shuffle(idxs)
    if take_rest:
        idxs = idxs[num:]
    else:
        idxs = idxs[:num]
    return idxs

class SinglePovertyMap(torch.utils.data.Dataset):
    
    _allowed_countries = [
        'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
        'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
        'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
        'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe'
    ]  # 23
    
    _allowed_areas = ['urban', 'rural']

    _BAND_ORDER = [
        'BLUE', 'GREEN', 'RED', 'SWIR1',
        'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS'
    ]

    _SURVEY_NAMES_2009_17A = {  # fold = A
        'train': ['cameroon', 'democratic_republic_of_congo', 'ghana', 'kenya',
                  'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
                  'togo', 'uganda', 'zambia', 'zimbabwe'],
        'ood_val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
        'ood_test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    }

    _SURVEY_NAMES_2009_17B = {  # fold = B
        'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
                  'ethiopia', 'kenya', 'lesotho', 'mali', 'mozambique',
                  'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
        'ood_val': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
        'ood_test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    }

    _SURVEY_NAMES_2009_17C = {  # fold = C
        'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
                  'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
                  'sierra_leone', 'tanzania', 'zambia'],
        'ood_val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
        'ood_test': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
    }

    _SURVEY_NAMES_2009_17D = {  # fold = D
        'train': ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
                  'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
                  'sierra_leone', 'tanzania', 'zimbabwe'],
        'ood_val': ['kenya', 'lesotho', 'senegal', 'zambia'],
        'ood_test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    }

    _SURVEY_NAMES_2009_17E = {  # fold = E
        'train': ['benin', 'burkina_faso', 'cameroon', 'democratic_republic_of_congo',
                  'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
                  'tanzania', 'togo', 'uganda', 'zimbabwe'],
        'ood_val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
        'ood_test': ['kenya', 'lesotho', 'senegal', 'zambia'],
    }

    def __init__(self,
                 root: str = 'data/wilds/poverty_v1.1',
                 country: typing.Union[str, int] = 'rwanda',  # 16
                 split: str = 'train',                        # train, id_val, id_test, ood_val, ood_test
                 random_state: int = 42,
                 in_memory: int = 0,
                 ) -> None:

        self.root = root; assert os.path.isdir(self.root)
        self.split = split; assert self.split in ['train', 'id_val', 'id_test', 'ood_val', 'ood_test', None]
        if isinstance(country, int):
            self.country: str = self._allowed_countries[country]
        else:
            self.country: str = country
        assert self.country in self._allowed_countries
        self.random_state = random_state
        self.in_memory = in_memory

        # TODO: remove if unused
        self._country2idx = {c: i for i, c in enumerate(self._allowed_countries)}
        self._idx2country = self._allowed_countries

        # load metadata
        metadata = pd.read_csv(os.path.join(self.root, 'dhs_metadata.csv'))
        metadata['area'] = metadata['urban'].apply(lambda b: 'urban' if b else 'rural')
        metadata['original_idx'] = metadata.index.values  # IMPORTANT; for input file configuration

        # keeps rows specific to country
        country_indices = np.where(metadata['country'] == self.country)[0]

        # keep rows specific to split
        if self.split not in ['train', 'id_val', 'id_test']:
            indices = country_indices
        else:
            N = int(len(country_indices) * 0.2)  # FIXME: use split provided by original wilds
            if self.split == 'train':
                indices = subsample_idxs(country_indices, take_rest=True, num=N, seed=random_state)
            elif self.split == 'id_val':
                indices = subsample_idxs(country_indices, take_rest=False, num=N, seed=random_state)
                indices = indices[N//2:]
            elif self.split == 'id_test':
                indices = subsample_idxs(country_indices, take_rest=False, num=N, seed=random_state)
                indices = indices[:N//2]
            else:
                indices = country_indices
        
        self.metadata = metadata.iloc[indices].reset_index(drop=True, inplace=False)

        # TODO: remove if unused
        self._true_indices = indices

        # list of input files (IMPORTANT)
        self.input_files: typing.List[str] = [
            os.path.join(
                self.root, f'images/landsat_poverty_img_{idx}.npz'
            ) for idx in self.metadata['original_idx'].values
        ]
        assert all([os.path.exists(f) for f in self.input_files])
        if self.in_memory > 0:
            raise NotImplementedError
        else:
            self.inputs = None

        # targets, domains, evaluation groups (IMPORTANT)
        self.targets = torch.from_numpy(self.metadata['wealthpooled'].values).float()
        self.domains = torch.tensor(
            [self._allowed_countries.index(c) for c in self.metadata['country'].values],
            dtype=torch.long,
        )
        self.eval_groups = torch.from_numpy(self.metadata['urban'].astype(int).values).long()  # 1 for urban, 0 for rural
        
    def get_input(self, index: int) -> torch.FloatTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            img: np.ndarray = np.load(self.input_files[index])['x']  # already np.float32
            return torch.from_numpy(img).float()

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

def get_country_splits(fold: str) -> typing.Dict[str, typing.List[str]]:
    if fold not in ['A', 'B', 'C', 'D', 'E']:
        raise ValueError
    survey_names: typing.Dict[str, dict] = {
        '2009-17A': SinglePovertyMap._SURVEY_NAMES_2009_17A,
        '2009-17B': SinglePovertyMap._SURVEY_NAMES_2009_17B,
        '2009-17C': SinglePovertyMap._SURVEY_NAMES_2009_17C,
        '2009-17D': SinglePovertyMap._SURVEY_NAMES_2009_17D,
        '2009-17E': SinglePovertyMap._SURVEY_NAMES_2009_17E,
    }
    return survey_names[f'2009-17{fold}']

class WildsPovertyMapDataset(MultipleDomainCollection):
    def __init__(self,
                 root: str = './data/wilds/poverty_v1.1',
                 train_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['train'],
                 validation_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_val'],
                 test_domains: typing.Iterable[str] = SinglePovertyMap._SURVEY_NAMES_2009_17A['ood_test'],
                 ) -> None:
        super(WildsPovertyMapDataset, self).__init__()
        # self.save_hyperparameters()
        self.root = root
        # dataloader arguments
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.prefetch_factor = prefetch_factor
        # # list of domain strings
        self._train_countries = train_domains#[0]
        self._validation_countries = validation_domains#[0]
        self._test_countries = test_domains#[0]
        # list of domain integers
        self.train_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._train_countries]
        self.validation_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._validation_countries]
        self.test_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._test_countries]

        # data_splits: dict = SinglePovertyMap.get_country_splits(fold=self.fold)
        # self._train_countries: typing.List[str] = data_splits['train']
        # self._validation_countries: typing.List[str] = data_splits['ood_val']
        # self._test_countries: typing.List[str] = data_splits['ood_test']
        # self.train_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._train_countries]
        # self.validation_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._validation_countries]
        # self.test_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._test_countries]

        # collection of datasets
        self._train_datasets = list()
        self._id_validation_datasets = list()
        self._ood_validation_datasets = list()
        self._id_test_datasets = list()
        self._ood_test_datasets = list()

        # TODO: change 'random_state' based on fold
        stage = None

        if (stage is None) or (stage == 'fit') or (stage == 'validate'):
            # (1) train / id-validation
            for c in self._train_countries:
                self._train_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='train', random_state=42)
                ]
                self._id_validation_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='id_val', random_state=42)
                ]

            # (2) ood_validation
            for c in self._validation_countries:
                self._ood_validation_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='ood_val', random_state=42)
                ]

        if (stage is None) or (stage == 'test'):

            # (3) ood_test
            for c in self._test_countries:
                self._ood_test_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='ood_test', random_state=42)
                ]
                self._test_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='ood_test', random_state=42)
                ]

            # (4) id_test
            for c in self._train_countries:
                self._id_test_datasets += [
                    SinglePovertyMap(root=self.root, country=c, split='id_test', random_state=42)
                ]

        # # Train & ID-validation sets
        # self._train_datasets, self._id_validation_datasets = list(), list()
        # for country in self._train_countries:
        #     if self.reserve_id_validation:
        #         self._train_datasets.append(
        #             SinglePovertyMap(root=root, country=country, 
        #                              fold=self.fold, split='train', in_memory=self.in_memory)
        #             )
        #         self._id_validation_datasets.append(
        #             SinglePovertyMap(root=root, country=country,
        #                              fold=self.fold, split='id_val', in_memory=self.in_memory)
        #             )
        #     else:
        #         self._train_datasets.append(
        #             SinglePovertyMap(root=root, country=country,
        #                              fold=self.fold, split=None, in_memory=self.in_memory)
        #             )

        # # OOD_validation sets
        # self._ood_validation_datasets = list()
        # for country in self._validation_countries:
        #     self._ood_validation_datasets.append(
        #         SinglePovertyMap(root=root, country=country,
        #                         fold=self.fold, split=None, in_memory=self.in_memory)
        #         )
        
        # # OOD_test sets
        # self._test_datasets = list()
        # for country in self._test_countries:
        #     self._test_datasets.append(
        #         SinglePovertyMap(root=root, country=country,
        #                             fold=self.fold, split=None, in_memory=self.in_memory)
        #         )

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (8, 224, 224)

    @property
    def num_classes(self) -> int:
        return 1
    
    def __repr__(self):
        _repr = (
            f"{self.__class__.__name__}\n",
            f"- {len(self.train_domains):,} training domains: {', '.join([f'{s}({i})' for s, i in zip(self._train_countries, self.train_domains)])}\n",
            f"- {len(self.validation_domains):,} validation domains: {', '.join([f'{s}({i})' for s, i in zip(self._validation_countries, self.validation_domains)])}\n",
            f"- {len(self.test_domains):,} test domains: {', '.join([f'{s}({i})' for s, i in zip(self._test_countries, self.test_domains)])}\n",
            # f"- Reserve ID-validation data: {self.reserve_id_validation}"
        )
        return ''.join(_repr)

class SinglePovertyMap_(torch.utils.data.Dataset):
    _split_scheme: str = 'countries'                              # TODO: remove as it is not used.
    _allowed_folds: typing.List[str] = ['A', 'B', 'C', 'D', 'E']  # folds is only used to split train set.
    _allowed_countries: typing.List[str] = [
        'angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
        'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
        'lesotho', 'malawi', 'mali', 'mozambique', 'nigeria', 'rwanda', 'senegal',
        'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe'
    ]
    _allowed_areas: typing.List[str] = ['urban', 'rural']
    _BAND_ORDER: typing.List[str] = [
        'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS'
    ]

    _SURVEY_NAMES_2009_17A = {
        'train': ['cameroon', 'democratic_republic_of_congo', 'ghana', 'kenya',
                  'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
                  'togo', 'uganda', 'zambia', 'zimbabwe'],
        'ood_val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
        'ood_test': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
    }
    _SURVEY_NAMES_2009_17B = {
        'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
                  'ethiopia', 'kenya', 'lesotho', 'mali', 'mozambique',
                  'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
        'ood_val': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
        'ood_test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    }
    _SURVEY_NAMES_2009_17C = {
        'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
                  'guinea', 'kenya', 'lesotho', 'mali', 'rwanda', 'senegal',
                  'sierra_leone', 'tanzania', 'zambia'],
        'ood_val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
        'ood_test': ['cameroon', 'ghana', 'malawi', 'zimbabwe'],
    }
    _SURVEY_NAMES_2009_17D = {
        'train': ['angola', 'benin', 'burkina_faso', 'cameroon', 'cote_d_ivoire',
                  'ethiopia', 'ghana', 'guinea', 'malawi', 'mali', 'rwanda',
                  'sierra_leone', 'tanzania', 'zimbabwe'],
        'ood_val': ['kenya', 'lesotho', 'senegal', 'zambia'],
        'ood_test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    }
    _SURVEY_NAMES_2009_17E = {
        'train': ['benin', 'burkina_faso', 'cameroon', 'democratic_republic_of_congo',
                  'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
                  'tanzania', 'togo', 'uganda', 'zimbabwe'],
        'ood_val': ['angola', 'cote_d_ivoire', 'ethiopia', 'mali', 'rwanda'],
        'ood_test': ['kenya', 'lesotho', 'senegal', 'zambia'],
    }
    def __init__(self,
                 root: str = 'data/wilds/poverty_v1.1',
                 country: str = None,
                 fold: str = 'A',
                 split: str = None,
                 in_memory: int = 0) -> None:

        # def __init__(self,
        #              root: str = 'data/wilds/poverty_v1.1',
        #              country: typing.Union[str, int] = 'rwanda',  # 16
        #              split: str = 'train',                        # train, id_val, id_test, ood_val, ood_test
        #              random_state: int = 42,
        #              in_memory: int = 0,
        #              ) -> None:

        self.root: str = root
        self.country: str = country
        self.fold: str = fold
        self.split: str = split
        self.in_memory: int = in_memory

        # here, `self.fold` is to check in which split `self.country` belongs to.
        if self.fold not in self._allowed_folds:
            raise ValueError(
                f"Invalid fold: {self.fold}. Use one of {', '.join(self._allowed_folds)}"
            )

        # a country defines a domain.
        if self.country not in self._allowed_countries:
            raise ValueError(
                f"Invalid country: {self.country}. Use one of {', '.join(self._allowed_countries)}"
            )

        # Find which data split (i.e. train, ood_val, ood_test) the specified country
        # belongs to, and check whether it matches the provided argument `country`.
        split_of_country: str = None
        data_splits: dict = self.get_country_splits(fold=self.fold)
        for split_name, countries in data_splits.items():
            if self.country in countries:
                split_of_country: str = split_name
                break
        
        if split_of_country == 'train':
            if self.split not in ['train', 'id_val', 'id_test']:
                raise ValueError(
                    f"Argument split={self.split} incompatible with {split_of_country} data."
                    )
        else:
            if self.split is not None:
                raise NotImplementedError(
                    f"Argument split={self.split} incompatible with {split_of_country} data.",
                    f"Provided arguments: country={self.country}, fold={self.fold}."
                    )

        # Read metadata: note that our implementation is different from the official code.
        # Please refer to `notebooks/07_wilds_explore.ipynb` for details.
        metadata = pd.read_csv(os.path.join(self.root, f'metadata_{self.fold}.csv'))
        metadata['area'] = metadata['urban'].apply(lambda b: 'urban' if b else 'rural')
        
        # TODO: add option to also use `area` to define domains.
        # Keep rows of metadata specific to 'country' & 'split' (done!)
        rows_to_keep = (metadata['country'] == self.country)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 'train')
        elif self.split == 'id_val':
            rows_to_keep = rows_to_keep & (metadata['split'] == 'id_val')
        elif self.split == 'id_test':
            rows_to_keep = rows_to_keep & (metadata['split'] == 'id_test')
        else:
            pass
        metadata = metadata.loc[rows_to_keep].copy()

        # Keep original index for input file configuration (as below)
        metadata['original_idx'] = metadata.index.values
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files: typing.List[str] = [
            os.path.join(
                self.root, f'images/landsat_poverty_img_{idx}.npz'
            ) for idx in metadata['original_idx'].values
        ]
        if self.in_memory > 0:
            raise NotImplementedError
        else:
            self.inputs = None

        # TODO: sanity check on target shape; (N, ) vs. (N, 1)
        self.targets = torch.from_numpy(metadata['wealthpooled'].values).float()
        self.domains = torch.tensor(
            [self._allowed_countries.index(c) for c in metadata['country'].values],
            dtype=torch.long,
        )
        self.eval_groups = torch.from_numpy(metadata['urban'].astype(int).values).long()  # 1 for urban, 0 for rural
        self.metadata = metadata  # save metadata, just in case...

    def get_input(self, index: int) -> torch.FloatTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            img: np.ndarray = np.load(self.input_files[index])['x']  # already np.float32
            return torch.from_numpy(img).float()

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    @classmethod
    def get_country_splits(cls, fold: str) -> typing.Dict[str, typing.List[str]]:
        if fold not in ['A', 'B', 'C', 'D', 'E']:
            raise ValueError
        survey_names: typing.Dict[str, dict] = {
            '2009-17A': cls._SURVEY_NAMES_2009_17A,
            '2009-17B': cls._SURVEY_NAMES_2009_17B,
            '2009-17C': cls._SURVEY_NAMES_2009_17C,
            '2009-17D': cls._SURVEY_NAMES_2009_17D,
            '2009-17E': cls._SURVEY_NAMES_2009_17E,
        }
        return survey_names[f'2009-17{fold}']

    @property
    def domain_indicator(self) -> int:
        return self._allowed_countries.index(self.country)

class WildsPovertyMapDataset_(MultipleDomainCollection):
    def __init__(self,
                 args,
                 reserve_id_validation: bool = True,
                 in_memory: int = 0) -> None:
        super(WildsPovertyMapDataset_, self).__init__()
        
        self.args = args
        # self.root = args.root
        # self.fold = args.fold
        self.reserve_id_validation: bool = reserve_id_validation
        self.in_memory: int = in_memory

        # data_splits: dict = SinglePovertyMap.get_country_splits(fold=self.args.fold)
        data_splits: dict = get_country_splits(fold=self.args.fold)

        self._train_countries: typing.List[str] = data_splits['train']
        self._validation_countries: typing.List[str] = data_splits['ood_val']
        self._test_countries: typing.List[str] = data_splits['ood_test']

        self.train_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._train_countries]
        self.validation_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._validation_countries]
        self.test_domains = [SinglePovertyMap._allowed_countries.index(c) for c in self._test_countries]

        # Train & ID-validation sets
        self._train_datasets, self._id_validation_datasets = list(), list()
        for country in self._train_countries:
            if self.reserve_id_validation:
                self._train_datasets.append(
                    SinglePovertyMap(root=self.args.root, country=country, split='train', in_memory=self.in_memory)
                    )
                self._id_validation_datasets.append(
                    SinglePovertyMap(root=self.args.root, country=country, split='id_val', in_memory=self.in_memory)
                    )
            else:
                self._train_datasets.append(
                    SinglePovertyMap(root=self.args.root, country=country, split=None, in_memory=self.in_memory)
                    )

        # OOD_validation sets
        self._ood_validation_datasets = list()
        for country in self._validation_countries:
            self._ood_validation_datasets.append(
                SinglePovertyMap(root=self.args.root, country=country, split=None, in_memory=self.in_memory)
                )
        
        # OOD_test sets
        self._test_datasets = list()
        for country in self._test_countries:
            self._test_datasets.append(
                SinglePovertyMap(root=self.args.root, country=country, split=None, in_memory=self.in_memory)
                )

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (8, 224, 224)

    @property
    def num_classes(self) -> int:
        return 1
    
    def __repr__(self):
        _repr = (
            f"{self.__class__.__name__}\n",
            f"- {len(self.train_domains):,} training domains: {', '.join([f'{s}({i})' for s, i in zip(self._train_countries, self.train_domains)])}\n",
            f"- {len(self.validation_domains):,} validation domains: {', '.join([f'{s}({i})' for s, i in zip(self._validation_countries, self.validation_domains)])}\n",
            f"- {len(self.test_domains):,} test domains: {', '.join([f'{s}({i})' for s, i in zip(self._test_countries, self.test_domains)])}\n",
            f"- Reserve ID-validation data: {self.reserve_id_validation}"
        )
        return ''.join(_repr)


class SingleRxRx1(torch.utils.data.Dataset):
    """
        Input (x); is a 3-channel image of cells obtained by ﬂuorescent microscopy (nuclei, endoplasmic reticuli, actin),
        Output (y); indicates which of the 1,139 genetic treatments (including no treatment) the cells received,
        Domain (d); speciﬁes the experimental batch of the image. (train:ood_val:test = 33:4:14), 51 in total.
            Size; train:ood_val:test = 40612:9854:34432 (we use the original `id_test` set as `id_val` of size 40612.)
    """
    _experiment_batches: typing.List[str] = [
        'HEPG2-01', 'HEPG2-02', 'HEPG2-03', 'HEPG2-04', 'HEPG2-05', 'HEPG2-06',
        'HEPG2-07', 'HEPG2-08', 'HEPG2-09', 'HEPG2-10', 'HEPG2-11',
        'HUVEC-01', 'HUVEC-02', 'HUVEC-03', 'HUVEC-04', 'HUVEC-05', 'HUVEC-06',
        'HUVEC-07', 'HUVEC-08', 'HUVEC-09', 'HUVEC-10', 'HUVEC-11', 'HUVEC-12',
        'HUVEC-13', 'HUVEC-14', 'HUVEC-15', 'HUVEC-16', 'HUVEC-17', 'HUVEC-18',
        'HUVEC-19', 'HUVEC-20', 'HUVEC-21', 'HUVEC-22', 'HUVEC-23', 'HUVEC-24',
        'RPE-01', 'RPE-02', 'RPE-03', 'RPE-04', 'RPE-05', 'RPE-06',
        'RPE-07', 'RPE-08', 'RPE-09', 'RPE-10', 'RPE-11',
        'U2OS-01', 'U2OS-02', 'U2OS-03', 'U2OS-04', 'U2OS-05',
    ]
    _valid_eval_groups: typing.List[str] = [
        'HEPG2', 'HUVEC', 'RPE', 'U2OS'
    ]
    def __init__(self,
                 root: str = 'data/wilds/rxrx1_v1.0',
                 experiment_batch: str = None,
                 split: str = None,
                 in_memory: int = 0) -> None:
        super(SingleRxRx1, self).__init__()

        self.root: str = root
        self.experiment_batch: str = experiment_batch
        self.split: str = split
        self.in_memory: int = in_memory

        if self.experiment_batch not in self._experiment_batches:
            raise IndexError(f'Invalid experiment batch: {self.experiment_batch}')

        if self.split is not None:
            if self.split not in ['train', 'id_val']:
                raise KeyError(f'Invalid split: {self.split}')

        # Read metadata
        metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'))

        # Use training data with `site=2` as id validation data
        id_val_mask: pd.Series = (metadata['dataset'] == 'train') & (metadata['site'] == 2)
        metadata.loc[id_val_mask, 'dataset'] = 'id_val'

        # Keep rows of metadata specific to `experiment_batch` & `split`
        # `train` & `id_val` data have overlapping `experiment_batch`es
        rows_to_keep = (metadata['experiment'] == experiment_batch)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['dataset'] == 'train')
        elif self.split == 'id_val':
            rows_to_keep = rows_to_keep & (metadata['dataset'] == 'id_val')
        else:
            pass
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files: typing.List[str] = metadata.apply(self._create_filepath, axis=1).to_list()
        if self.in_memory > 0:
            start = time.time()
            print(f'Loading {len(self.input_files):,} images in memory (experiment={self.experiment_batch}, split={self.split}).', end=' ')
            self.inputs = self.load_images(self.input_files, p=self.in_memory, as_tensor=True)
            print(f'Elapsed Time: {time.time() - start:.2f} seconds.')
        else:
            self.inputs = None
        
        self.targets = torch.from_numpy(metadata['sirna_id'].values).long()
        self.domains = torch.LongTensor(
            [self._experiment_batches.index(s) for s in metadata['experiment'].values]
        )
        self.eval_groups = [self._valid_eval_groups.index(str(c)) for c in metadata['cell_type'].values]
        self.eval_groups = torch.from_numpy(np.array(self.eval_groups)).long()
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            # 3d RGB tensor, uint8 (0-255)
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def _create_filepath(self, row: pd.Series):
        """Create filepath from row of metadata dataframe."""
        path_strings: typing.List[str] = [
            "images", row['experiment'], f"Plate{row['plate']}", f"{row['well']}_s{row['site']}.png",
        ]
        return os.path.join(self.root, *path_strings)

    @staticmethod
    def load_images(filenames: typing.List[str],
                    p: int,
                    as_tensor: bool = True,
                    ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        """
        Load images with multiprocessing if p > 0.
        Arguments:
            filenames: list of filename strings.
            p: int for number of cpu threads to use for data loading.
        """
        with RayPool(processes=p) as pool:
            images = pool.map(functools.partial(read_image, mode=ImageReadMode.RGB), filenames)
            pool.close(); pool.join(); time.sleep(5.0)

        return torch.stack(images, dim=0) if as_tensor else images

    @property
    def domains_as_strings(self) -> typing.List[str]:
        return self._experiment_batches

    @property
    def domain_indicator(self) -> int:
        return self._experiment_batches.index(self.experiment_batch)


class WildsRxRx1Dataset(MultipleDomainCollection):
    
    _train_domains_str: typing.List[str] = [
        'HEPG2-01', 'HEPG2-02', 'HEPG2-03', 'HEPG2-04', 'HEPG2-05', 'HEPG2-06',
        'HEPG2-07', 'HUVEC-01', 'HUVEC-02', 'HUVEC-03', 'HUVEC-04', 'HUVEC-05',
        'HUVEC-06', 'HUVEC-07', 'HUVEC-08', 'HUVEC-09', 'HUVEC-10', 'HUVEC-11',
        'HUVEC-12', 'HUVEC-13', 'HUVEC-14', 'HUVEC-15', 'HUVEC-16',
        'RPE-01', 'RPE-02', 'RPE-03', 'RPE-04', 'RPE-05', 'RPE-06', 'RPE-07',
        'U2OS-01', 'U2OS-02', 'U2OS-03',
    ]
    _validation_domains_str: typing.List[str] = [
        'HEPG2-08', 'HUVEC-17', 'RPE-08', 'U2OS-04',
    ]
    _test_domains_str: typing.List[str] = [
        'HEPG2-09', 'HEPG2-10', 'HEPG2-11', 'HUVEC-18', 'HUVEC-19',
        'HUVEC-20', 'HUVEC-21', 'HUVEC-22', 'HUVEC-23', 'HUVEC-24',
        'RPE-09', 'RPE-10', 'RPE-11', 'U2OS-05',
    ]
    
    def __init__(self,
                 root: str = './data/benchmark/rxrx1_v1.0',
                 reserve_id_validation: bool = True,
                 in_memory: int = 0) -> None:
        super(WildsRxRx1Dataset, self).__init__()

        self.reserve_id_validation: bool = reserve_id_validation
        self.in_memory: int = in_memory

        # Train & ID-validation sets
        self._train_datasets, self._id_validation_datasets = list(), list()
        for domain in self._train_domains_str:
            if self.reserve_id_validation:
                self._train_datasets.append(
                    SingleRxRx1(root=root, experiment_batch=domain, split='train', in_memory=self.in_memory)
                )
                self._id_validation_datasets.append(
                    SingleRxRx1(root=root, experiment_batch=domain, split='id_val', in_memory=self.in_memory)
                )
            else:
                self._train_datasets.append(
                    SingleRxRx1(root=root, experiment_batch=domain, split=None, in_memory=self.in_memory)
                )

        # OOD_validation sets
        self._ood_validation_datasets = list()
        for domain in self._validation_domains_str:
            self._ood_validation_datasets.append(
                SingleRxRx1(root=root, experiment_batch=domain, split=None, in_memory=self.in_memory)
            )

        # OOD_test sets
        self._test_datasets = list()
        for domain in self._test_domains_str:
            self._test_datasets.append(
                SingleRxRx1(root=root, experiment_batch=domain, split=None, in_memory=self.in_memory)
            )

        self.train_domains: typing.List[int] = [dset.domain_indicator for dset in self._train_datasets]
        self.validation_domains: typing.List[int] = [dset.domain_indicator for dset in self._ood_validation_datasets]
        self.test_domains: typing.List[int] = [dset.domain_indicator for dset in self._test_datasets]

    @property
    def input_shape(self) -> typing.Tuple[int]:
        # Refer to Section E.3.4 for details on data processing
        return (3, 256, 256)

    @property
    def num_classes(self) -> int:
        return 1139

    def __repr__(self) -> str:
        _repr = (
            f"{self.__class__.__name__}\n",
            f"- Training domains: {', '.join([f'{s}({i})' for s, i in zip(self._train_domains_str, self.train_domains)])}\n",
            f"- Validation domains: {', '.join([f'{s}({i})' for s, i in zip(self._validation_domains_str, self.validation_domains)])}\n",
            f"- Test domains: {', '.join([f'{s}({i})' for s, i in zip(self._test_domains_str, self.test_domains)])}\n",
            f"- Reserve ID-validation data: {self.reserve_id_validation}"
        )
        return ''.join(_repr)

class SingleCamelyon(torch.utils.data.Dataset):
    _allowed_hospitals = [0, 1, 2, 3, 4]
    def __init__(self,
                 root: str = 'data/wilds/camelyon17_v1.0',
                 hospital: int = 0,
                 split: str = None,
                 in_memory: int = 0) -> None:
        super(SingleCamelyon, self).__init__()

        self.root: str = root
        self.hospital: int = hospital
        self.split: str = split
        self.in_memory: int = in_memory

        if self.hospital not in self._allowed_hospitals:
            raise IndexError

        if self.split is not None:
            if self.split not in ['train', 'val']:
                raise KeyError

        # Read metadata
        metadata = pd.read_csv(
            os.path.join(self.root, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'}
        )

        # Keep rows of metadata specific to `hospital` & `split`
        rows_to_keep = (metadata['center'] == hospital)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 0)  # train: 0
        elif self.split == 'val':
            rows_to_keep = rows_to_keep & (metadata['split'] == 1)  # val: 1
        else:
            pass
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files: list = [
            os.path.join(
                self.root,
                f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
            ) for patient, node, x, y in
            metadata.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)
        ]
        if self.in_memory > 0:
            start = time.time()
            print(f'Loading {len(self.input_files):,} images in memory (hospital={self.hospital}, split={self.split}).', end=' ')
            self.inputs = self.load_images(self.input_files, p=self.in_memory, as_tensor=True)
            print(f'Elapsed Time: {time.time() - start:.2f} seconds.')
        else:
            self.inputs = None
        self.targets = torch.LongTensor(metadata['tumor'].values)
        self.domains = torch.LongTensor(metadata['center'].values)
        self.eval_groups = torch.LongTensor(metadata['slide'].values)
        self.metadata = metadata

    def get_input(self, index: int) -> torch.ByteTensor:
        if self.inputs is not None:
            return self.inputs[index]
        else:
            return read_image(self.input_files[index], mode=ImageReadMode.RGB)

    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]

    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def load_images(filenames: typing.List[str],
                    p: int,
                    as_tensor: bool = True,
                    ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        """
        Load images with multiprocessing if p > 0.
        Arguments:
            filenames: list of filename strings.
            p: int for number of cpu threads to use for data loading.
            as_tensor: bool, returns a stacked tensor if True, a list of tensor images if False.
        Returns:
            ...
        """
        with RayPool(processes=p) as pool:
            images = pool.map(functools.partial(read_image, mode=ImageReadMode.RGB), filenames)
            pool.close(); pool.join(); time.sleep(5.0)

        return torch.stack(images, dim=0) if as_tensor else images

    @property
    def domain_indicator(self) -> int:
        return self.hospital

class WildsCamelyonDataset(MultipleDomainCollection):
    def __init__(self,
                 root: str = './data/wilds/camelyon17_v1.0',
                 train_domains: typing.Union[int, typing.List[int]] = [0, 3, 4],
                 validation_domains: typing.Union[int, typing.List[int], None] = [1],
                 test_domains: typing.Union[int, typing.List[int], None] = [2],
                 reserve_id_validation: bool = True,
                 in_memory: int = 0) -> None:
        super(WildsCamelyonDataset, self).__init__()

        self.train_domains = self.as_list(train_domains)
        self.validation_domains = self.as_list(validation_domains)
        self.test_domains = self.as_list(test_domains)
        self.reserve_id_validation: bool = reserve_id_validation
        self.in_memory: int = in_memory

        # Train & ID-validation sets
        self._train_datasets, self._id_validation_datasets = list(), list()
        for domain in self.train_domains:
            if self.reserve_id_validation:
                self._train_datasets.append(
                    SingleCamelyon(root=root, hospital=domain, split='train', in_memory=self.in_memory)
                )
                self._id_validation_datasets.append(
                    SingleCamelyon(root=root, hospital=domain, split='val', in_memory=self.in_memory)
                )
            else:
                self._train_datasets.append(
                    SingleCamelyon(root=root, hospital=domain, split=None, in_memory=self.in_memory)
                )

        # OOD_validation sets
        self._ood_validation_datasets = list()
        for domain in self.validation_domains:
            self._ood_validation_datasets.append(
                SingleCamelyon(root=root, hospital=domain, split=None, in_memory=self.in_memory)
            )

        # Test sets
        self._test_datasets = list()
        for domain in self.test_domains:
            self._test_datasets.append(
                SingleCamelyon(root=root, hospital=domain, split=None, in_memory=self.in_memory)
            )

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, 96, 96)

    @property
    def num_classes(self) -> int:
        return 2

    def __repr__(self):
        _repr = (
            f"{self.__class__.__name__}\n",
            f"- Training domains: {self.train_domains}\n",
            f"- Validation domains: {self.validation_domains}\n",
            f"- Test domains: {self.test_domains}\n",
            f"- Reserve ID-validation data: {self.reserve_id_validation}"
        )
        return ''.join(_repr)

class SingleIWildCam(torch.utils.data.Dataset):
    """
    Input (x): RGB images from camera traps
    Label (y): one of 186 classes corresponding to animal species
        In the metadata, each instance is annotated with the ID of the location
            (camera trap) it came from.
    """
    size: tuple = (448, 448)
    def __init__(self,
                 root: str = './data/wilds/iwildcam_v2.0',
                 location: int = None,
                 split: str = None,
                 metadata: pd.DataFrame = None,
                 use_id_test: bool = True,
                 in_memory: bool = False,
                 ):
        super(SingleIWildCam, self).__init__()
        
        self.root: str = root
        self.location: int = location
        self.split: str = split
        self.use_id_test: bool = use_id_test
        self.in_memory: bool = in_memory

        # load metadata
        if not isinstance(metadata, pd.DataFrame):    
            metadata = pd.read_csv(
                os.path.join(self.root, 'metadata.csv'), index_col=0
            )
        else:
            assert all(
                [c in metadata.columns for c in ['split', 'location_remapped', 'y', 'filename']]
            )

        assert isinstance(self.location, int)
        if self.split is not None:
            if self.split not in ['train', 'id_val']:
                raise KeyError(f'Invalid split: {self.split}')
        
        # Keep rows of metadata specific to `location_remapped` and `split`
        rows_to_keep = (metadata['location_remapped'] == self.location)
        if self.split == 'train':
            rows_to_keep = rows_to_keep & (metadata['split'] == 'train')
        elif self.split == 'id_val':
            if self.use_id_test:
                rows_to_keep = rows_to_keep & metadata['split'].isin(['id_val', 'id_test'])
            else:
                rows_to_keep = rows_to_keep & (metadata['split'] == 'id_val')
        else:
            pass
        metadata = metadata.loc[rows_to_keep].copy()
        metadata = metadata.reset_index(drop=True, inplace=False)

        # Main attributes
        self.input_files: typing.List[str] = [
            os.path.join(self.root, 'train', filename) for filename in metadata['filename'].values
        ]  # all data instances are stored in the `train` folder
        if self.in_memory:
            raise NotImplementedError("Work in progress...")
        self.targets = torch.from_numpy(metadata['y'].values).long()
        self.domains = torch.from_numpy(metadata['location_remapped'].values).long()
        torch.from_numpy(metadata['location_remapped'].values).long()
        self.subdomains = torch.from_numpy(metadata['location_remapped'].values).long()
        
        self.eval_groups = self.domains.clone()
        self.metadata = metadata

        # FIXME; 
        from torchvision.transforms import Resize
        self.resize_fn = Resize(size=self.size)

    def get_input(self, index: int) -> torch.ByteTensor:
        img = read_image(self.input_files[index], mode=ImageReadMode.RGB)
        return self.resize_fn(img)
    
    def get_target(self, index: int) -> torch.LongTensor:
        return self.targets[index]

    def get_domain(self, index: int) -> torch.LongTensor:
        return self.domains[index]
    
    def get_eval_group(self, index: int) -> torch.LongTensor:
        return self.eval_groups[index]
    
    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.get_input(index),
            y=self.get_target(index),
            domain=self.get_domain(index),
            eval_group=self.get_eval_group(index),
        )

    def __len__(self) -> int:
        return len(self.metadata)

    @property
    def domain_indicator(self) -> int:
        return self.location


class WildsIWildCamDataset(MultipleDomainCollection):
    _domain_col: str = 'location_remapped'
    def __init__(self,
                 root: str = './data/benchmark/wilds/iwildcam_v2.0',
                 use_id_test: bool = True, ) -> None:
        super(WildsIWildCamDataset, self).__init__()

        self.root: str = root
        self.use_id_test: bool = use_id_test

        # load metadata
        metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'), index_col=0)

        # find domains (train)
        train_mask = (metadata['split'] == 'train')
        self.train_domains: list = sorted(metadata.loc[train_mask, self._domain_col].unique().tolist())

        # find domains (ID-val)
        if self.use_id_test:
            id_val_mask = metadata['split'].isin(['id_val', 'id_test'])
        else:
            id_val_mask = (metadata['split'] == 'id_val')
        self.id_validation_domains: list = sorted(metadata.loc[id_val_mask, self._domain_col].unique().tolist())

        # find domains (OOD-val)
        ood_val_mask = (metadata['split'] == 'val')
        self.validation_domains: list = sorted(metadata.loc[ood_val_mask, self._domain_col].unique().tolist())
        self.ood_validation_domains = self.validation_domains  # alias

        # find domains (test)
        test_mask = (metadata['split'] == 'test')
        self.test_domains: list = sorted(metadata.loc[test_mask, self._domain_col].unique().tolist())

        # (sanity check) are id validation domains a subset of the training domains?
        if not all([c in self.train_domains for c in self.id_validation_domains]):
            raise AssertionError
        # (sanity check) are train / ood validation / test domains mutually exclusive?
        if not all([c not in self.train_domains for c in self.validation_domains]):
            raise AssertionError
        if not all([c not in self.train_domains for c in self.test_domains]):
            raise AssertionError
        if not all([c not in self.test_domains for c in self.validation_domains]):
            raise AssertionError
        print(f"Domains are valid!")
        
        # Train datasets
        from rich.progress import track
        self._train_datasets = list()
        for l in track(self.train_domains, total=len(self.train_domains), description='Loading training data...'):
            self._train_datasets.append(
                SingleIWildCam(root=self.root, location=l, split='train')
            )
        assert all([isinstance(d, torch.utils.data.Dataset) for d in self._train_datasets])
        print(f"Loaded {len(self._train_datasets):,} training datasets.")
        
        self._id_validation_datasets = list()
        for l in track(self.id_validation_domains, total=len(self.id_validation_domains), description='Loading ID validation data...'):
            self._id_validation_datasets.append(
                SingleIWildCam(root=self.root, location=l, split='id_val', use_id_test=self.use_id_test)
            )
        assert all([isinstance(d, torch.utils.data.Dataset) for d in self._id_validation_datasets])
        print(f"Loaded {len(self._id_validation_datasets):,} ID validation datasets.")
        
        # OOD-validation datasets
        self._ood_validation_datasets = list()
        for l in track(self.validation_domains, total=len(self.validation_domains), description='Loading OOD validation data...'):
            self._ood_validation_datasets.append(
                SingleIWildCam(root=self.root, location=l, split=None)
            )
        assert all([isinstance(d, torch.utils.data.Dataset) for d in self._ood_validation_datasets])
        print(f"Loaded {len(self._ood_validation_datasets):,} OOD validation datasets.")
        
        # Test datasets
        self._test_datasets = list()
        for l in track(self.test_domains, total=len(self.test_domains), description='Loading test data...'):
            self._test_datasets.append(
                SingleIWildCam(root=self.root, location=l, split=None)
            )
        assert all([isinstance(d, torch.utils.data.Dataset) for d in self._test_datasets])
        print(f"Loaded {len(self._test_datasets):,} test datasets.")

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, 448, 448)  # original images are reshaped

    @property
    def num_classes(self) -> int:
        return 182

    def __repr__(self) -> str:
        _repr = (
            f'{self.__class__.__name__}\n',
            f'- Number of classes: {self.num_classes:,}\n',
             '- Resized input shape: ({}, {}, {})\n'.format(*self.input_shape),
            f'- Training domains: {self.train_domains}\n',
            f'- ID-validation domains: {self.id_validation_domains}\n',
            f'- OOD-validation domains: {self.validation_domains}\n',
            f'- Test domains: {self.test_domains}\n',
            f'- Add ID-test to ID-validation: {self.use_id_test}'
        )
        return ''.join(_repr)
