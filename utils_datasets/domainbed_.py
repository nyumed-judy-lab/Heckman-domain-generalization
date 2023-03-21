
import os
import time
import glob
import typing
import pathlib

import gdown
import tarfile

import numpy as np
import torch

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import Resize
from torchvision.transforms.functional import pil_to_tensor

from utils_datasets.base import MultipleDomainCollection
from utils_datasets.utils import DictionaryDataset


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")



class SingleVLCS(torch.utils.data.Dataset):

    _allowed_labels = ('bird', 'car', 'chair', 'dog', 'person')
    _allowed_domains = ('VOC2007', 'LabelMe', 'Caltech101', 'SUN09')
    _size = (224, 224)
    
    def __init__(self, input_files: np.ndarray):
        
        super(SingleVLCS, self).__init__()        
        self.input_files = input_files
        self.resize_fn = Resize(self._size)

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        
        filename: str = self.input_files[index]
        try:
            img = read_image(filename, mode=ImageReadMode.RGB)
        except RuntimeError as _:
            img = pil_loader(filename)
            img = pil_to_tensor(img)
        
        return dict(
            x=self.resize_fn(img),
            y=self._allowed_labels.index(pathlib.Path(filename).parent.name),
            domain=self._allowed_domains.index(pathlib.Path(filename).parent.parent.name),
            eval_group=0,
        )

    def __len__(self) -> int:
        return len(self.input_files)

class VLCS(MultipleDomainCollection):
    
    _url: str = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    _env_mapper = {
        'V': ('VOC2007', 0),
        'L': ('LabelMe', 1),
        'C': ('Caltech101', 2),
        'S': ('SUN09', 3)
    }
    
    def __init__(self,
                 root: str = 'data/domainbed/vlcs/',
                 train_environments: typing.List[str] = ['V', 'L', 'C'],
                 test_environments: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2,
                 download: bool = False):
        super(VLCS, self).__init__()
        
        self.root: str = root
        self.train_environments = train_environments
        self.test_environments = test_environments
        self.holdout_fraction: float = holdout_fraction
        
        if download:
            self._download()
        
        # find all JPG files
        input_files = np.array(glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True))

        # find environment names
        env_strings = np.array([pathlib.Path(f).parent.parent.name for f in input_files])
        
        # create {train, val} datasets for each domain
        self._train_datasets, self._id_validation_datasets = list(), list()
        for env in self.train_environments:
            
            # domain mask (indices)
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            env_indices = np.where(env_mask)[0]
            
            # train, validation mask (indices)
            np.random.shuffle(env_indices)
            split_idx = int(self.holdout_fraction * len(env_indices))
            val_indices = env_indices[:split_idx]
            train_indices = env_indices[split_idx:]

            self._train_datasets += [SingleVLCS(input_files[train_indices])]
            self._id_validation_datasets += [SingleVLCS(input_files[val_indices])]
        
        self._ood_validation_datasets = self._id_validation_datasets ##### TRICK for VLCS ID validationd

        # create test dataset
        self._test_datasets = list()
        for env in self.test_environments:

            # domain mask
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)

            self._test_datasets += [SingleVLCS(input_files[env_mask])]

        self.train_domains = [self._env_mapper[env][1] for env in self.train_environments]
        self.test_domains = [self._env_mapper[env][1] for env in self.test_environments]

    def _download(self):
        
        os.makedirs(self.root, exist_ok=True)
        _dst = os.path.join(self.root, 'VLCS.tar.gz')
        if not os.path.exists(_dst):
            gdown.download(self._url, _dst, quiet=False)
        
        tar = tarfile.open(_dst, "r:gz")
        tar.extractall(os.path.dirname(_dst))
        tar.close()

class DomainVLCS(MultipleDomainCollection):
    
    _url: str = "https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"
    _env_mapper = {
        0: ('VOC2007', 0),
        1: ('LabelMe', 1),
        2: ('Caltech101', 2),
        3: ('SUN09', 3)
    }
    
    def __init__(self,
                 root: str = 'data/domainbed/vlcs/',
                 train_domains: typing.List[str] = [0, 1],
                 validation_domains: typing.List[str] =[2],
                 test_domains: typing.List[str] = [3],
                #  train_domains: typing.List[str] = ['V', 'L'],
                #  validation_domains: typing.List[str] = ['C'],
                #  test_domains: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2):
        super(DomainVLCS, self).__init__()
        
        self.root: str = root
        self.train_domains = train_domains
        self.validation_domains = validation_domains ##### OOD Dataset
        self.test_domains = test_domains
        self.holdout_fraction: float = holdout_fraction


        
        # find all JPG files
        input_files = np.array(glob.glob(os.path.join(self.root, "**/*.jpg"), recursive=True))

        # find environment names
        env_strings = np.array([pathlib.Path(f).parent.parent.name for f in input_files])
        
        ########## Train & ID-validation sets
        # create {train, val} datasets for each domain
        self._train_datasets, self._id_validation_datasets = list(), list()
        for env in self.train_domains:
            
            # domain mask (indices)
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            env_indices = np.where(env_mask)[0]
            
            # train, validation mask (indices)
            np.random.shuffle(env_indices)
            split_idx = int(self.holdout_fraction * len(env_indices))
            val_indices = env_indices[:split_idx]
            train_indices = env_indices[split_idx:]

            self._train_datasets += [SingleVLCS(input_files[train_indices])]
            self._id_validation_datasets += [SingleVLCS(input_files[val_indices])]
            
        # ########## Train & ID-validation sets
        # # create {train, val} datasets for each domain
        
        # self._train_datasets = list()
        # for env in self.train_domains:
        #     # domain mask
        #     env_str, _ = self._env_mapper[env]
        #     env_mask = (env_strings == env_str)
        #     self._train_datasets += [SingleVLCS(input_files[env_mask])]

        #################### OOD validation dataset
        # create valid dataset 
        self._ood_validation_datasets = list()
        for env in self.validation_domains:
            # domain mask
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            self._ood_validation_datasets += [SingleVLCS(input_files[env_mask])]

        # create test dataset
        self._test_datasets = list()
        for env in self.test_domains:
            # domain mask
            env_str, _ = self._env_mapper[env]
            env_mask = (env_strings == env_str)
            self._test_datasets += [SingleVLCS(input_files[env_mask])]

        self.train_domains = [self._env_mapper[env][1] for env in self.train_domains]
        self.validation_domains = [self._env_mapper[env][1] for env in self.validation_domains] ##########
        self.test_domains = [self._env_mapper[env][1] for env in self.test_domains]
        # self.train_domains = [self._env_mapper[env][1] for env in self.train_environments]
        # self.validation_domains = [self._env_mapper[env][1] for env in self.valid_environments] ##########
        # self.test_domains = [self._env_mapper[env][1] for env in self.test_environments]



class PACS(MultipleDomainCollection):

    _url = "https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd"

    def __init__(self,
                 root: str = 'data/domainbed/pacs/',
                 train_environments: typing.List[str] = ['P', 'A', 'C'],
                 test_environments: typing.List[str] = ['S'],
                 holdout_fraction: float = 0.2,
                 download: bool = False):
        super(PACS, self).__init__()

        self.root: str = root

        if download:
            self._download()

    def _download(self) -> None:
        
        os.makedirs(self.root, exist_ok=True)
        
        # download
        _dst = os.path.join(self.root, 'PACS.zip')
        if not os.path.exists(_dst):
            gdown.download(self._url, _dst, quiet=False)
        
        # extract
        from zipfile import ZipFile
        zf = ZipFile(_dst, "r")
        zf.extractall(os.path.dirname(_dst))
        zf.close()
        
        # rename directory
        if os.path.isdir(os.path.join(self.root, 'kfold')):
            os.rename(
                src=os.path.join(self.root, "kfold"),
                dst=os.path.join(self.root, 'PACS')
            )


class ColoredMNIST(MultipleDomainCollection):
    _label_noise: float = 0.25
    _domain2cfg: dict = {
        0: 0.1,
        1: 0.2,
        2: 0.9,
    }
    def __init__(self,
                 train_domains: typing.List[float] = [0, 1, ], 
                 test_domains: typing.List[float] = [2, ],
                 holdout_fraction: typing.Optional[float] = 0.2,
                 mnist_root: typing.Optional[str] = './data/mnist/'):
        """
        Note that the input dimension is different from the DomainBed implementation.
            Ours: [70000, 3, 28, 28] vs. DomainBed: [70000, 2, 28, 28]
        """
        super(ColoredMNIST, self).__init__()
        
        # base attributes;
        self.train_domains: list = train_domains
        self.test_domains: list = test_domains
        self.holdout_fraction: float = holdout_fraction  # not necessary

        # load original mnist data;
        original_images, original_labels = self.load_mnist_data(root=mnist_root)
        
        # shuffle orders;
        shuffle = torch.randperm(len(original_labels), )
        original_images, original_labels = original_images[shuffle], original_labels[shuffle]

        # create colored mnist data
        domains: typing.List[float] = self.train_domains + self.test_domains  # joining two lists
        num_domains: int = len(domains)
        for i in range(num_domains):
            images, labels = self._make_cmnist_data(
                images=original_images[i::num_domains], labels=original_labels[i::num_domains],
                e=self._domain2cfg[domains[i]],
            )
            dataset = DictionaryDataset(
                {
                    'x': images,
                    'y': labels,
                    'domain': torch.full_like(labels, fill_value=domains[i]),
                    'eval_group': torch.full_like(labels, fill_value=-1),  # for consistency    
                }
            )
            if i < len(self.train_domains):
                # store `train` & `id_val` data separately
                split = int(len(labels) * holdout_fraction)
                random_indices = torch.randperm(len(labels), )
                id_val_indices, train_indices = random_indices[:split], random_indices[split:]
                self._train_datasets += [Subset(dataset, indices=train_indices)]
                self._id_validation_datasets += [Subset(dataset, indices=id_val_indices)]
            else:
                # store `test` data
                self._test_datasets += [dataset]

    def _make_cmnist_data(self, images: torch.FloatTensor, labels: torch.LongTensor, e: float):
        """
        Create CMNIST data where color is spuriously correlated with the label.
        Arguments: 
            images: 3d tensor of original mnist images; [N, H, W]
            labels: 1d tensor of original mnist labels; [N,  ]
            e: float \in (0, 1)
        """

        labels = (labels < 5).float()  # {0, 1}
        labels = self.torch_xor_(      # noisy labels by flipping with 0.25 probability
            labels, self.torch_bernoulli_(p=self._label_noise, size=(len(labels), )),
        )
        colors = self.torch_xor_(      # domain correlates with label
            labels, self.torch_bernoulli_(p=e, size=(len(labels), )),
        )

        images = torch.stack([images, images, images], dim=1)              # [N, 3, 28, 28]
        images[torch.arange(len(images)), (1 - colors).long(), :, :] *= 0  # 0 if R, 1 if G
        images[torch.arange(len(images)),                   2, :, :] *= 0  # 2 for B

        images = images.float().div(255.0)  # normalize; [0, 1] <- [0, 255]
        labels = labels.view(-1).long()     # integer labels

        return images, labels

    @property
    def input_shape(self) -> typing.Tuple[int]:
        return (3, 28, 28)

    @property
    def num_classes(self) -> int:
        return 2

    def __repr__(self):
        _repr: str = ''
        return _repr

    @staticmethod
    def load_mnist_data(root: str = './data/mnist/') -> typing.Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Load full MNIST data (both train & test).
        Arguments:
            root: (str) path to directory where `*.pt` files exist.
        Returns:
            a tuple of (images, labels) where
                images: [70000, 28, 28]
                labels: [70000, ]
        """
        
        mnist_train = MNIST(root=root, train=True)
        mnist_test = MNIST(root=root, train=False)
        images = torch.cat([mnist_train.data, mnist_test.data], dim=0)
        labels = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)

        return images, labels

    @staticmethod
    def torch_bernoulli_(p: float, size: tuple) -> torch.FloatTensor:
        return (torch.rand(size) < p).float()

    @staticmethod
    def torch_xor_(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a - b).abs()
