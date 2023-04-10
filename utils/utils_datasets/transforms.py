import typing
import collections

import torch
import torch.nn as nn
import torchvision.transforms as T

from torchvision.transforms.functional import rotate


class IdentityTransform(nn.Module):
    """
    This class exists for consistency. Please note with caution that
    Input transformations of text data is handled within each `torch.utils.data.Dataset`.
    """
    def __init__(self, **kwargs):
        super(IdentityTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class CutoutOnFloat(nn.Module):
    def __init__(self, div_factor: float = 2., **kwargs):
        super(CutoutOnFloat, self).__init__()
        self.div_factor: float = div_factor

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        assert (x.ndim == 4) and (x.shape[2] == x.shape[3])
        _, _, _, width = x.shape
        cutout_width = self._sample_uniform_single(0., int(width / self.div_factor))
        cutout_center_x = self._sample_uniform_single(0., width)
        cutout_center_y = self._sample_uniform_single(0., width)

        x0 = int(max(0, cutout_center_x - cutout_width / 2))
        y0 = int(max(0, cutout_center_y - cutout_width / 2))
        x1 = int(min(width, cutout_center_x + cutout_width / 2))
        y1 = int(min(width, cutout_center_y + cutout_width / 2))

        # fill with zeros
        x[:, :, x0:x1, y0:y1] = 0.
        return x

    def _new_forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        
        assert (x.ndim == 4) and (x.shape[-2] == x.shape[-1])  # equal height & width
        b, _, h, w = x.shape

        max_width = int(w / self.div_factor)
        cutout_widths = self._sample_uniform_(b, min_=0., max=max_width)
        cutout_center_x = list()
        for c_width in cutout_widths:
            cutout_center_x += []
        cutout_center_y = list()

        raise NotImplementedError

    @staticmethod
    def _sample_uniform_single(a: float, b: float) -> float:
        return torch.empty(1).uniform_(a, b).item()

    @staticmethod
    def _sample_uniform_(*size, min_: float, max_: float) -> torch.FloatTensor:
        return torch.zeros(*size).uniform_(min_, max_).flatten()


class SamplewiseNormalize(nn.Module):
    def __init__(self):
        super(SamplewiseNormalize, self).__init__()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim != 4:
            raise ValueError("Expects a 4-d tensor with shape (B, C, H, W).")
        mean = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        std = x.std(dim=(2, 3), keepdim=True)    # (B, C, 1, 1)
        std[std == 0.] = 1.                      # prevent zero division errors
        return  x.sub(mean).div(std)


class RandomRotation90(nn.Module):
    def __init__(self, interpolation=T.InterpolationMode.NEAREST):
        super(RandomRotation90, self).__init__()
        self.angles: list = [0, 90, 180, 270]
        self.interpolation = interpolation

    def forward(self, x: typing.Union[torch.ByteTensor, torch.FloatTensor]):
        angle = self.angles[torch.randint(low=0, high=len(self.angles), size=(1, ))]
        if angle > 0:
            x = rotate(img=x, angle=angle, interpolation=self.interpolation)
        return x


class RotatedMNISTTransform(nn.Module):
    def __init__(self, **kwargs):
        super(RotatedMNISTTransform, self).__init__()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x


class ColoredMNISTTransform(nn.Module):
    def __init__(self, **kwargs):
        super(ColoredMNISTTransform, self).__init__()
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x


class Camelyon17Transform(nn.Module):
    means: typing.Dict[int, typing.Tuple[float]] = {
        0: (0.735, 0.604, 0.701),
        1: (0.610, 0.461, 0.593),
        2: (0.673, 0.483, 0.739),
        3: (0.686, 0.490, 0.620),
        4: (0.800, 0.672, 0.820),
    }
    stds: typing.Dict[int, typing.Tuple[float]] = {
        0: (0.184, 0.220, 0.170),
        1: (0.186, 0.219, 0.174),
        2: (0.208, 0.235, 0.133),
        3: (0.202, 0.222, 0.173),
        4: (0.130, 0.159, 0.104),
    }
    size: typing.Tuple[int] = (96, 96)
    
    def __init__(self,
                 mean: typing.Tuple[float] = (0.720, 0.560, 0.715),
                 std: typing.Tuple[float] = (0.190, 0.224, 0.170),
                 augmentation: bool = False,
                 randaugment: bool = False,
                 **kwargs):
        super(Camelyon17Transform, self).__init__()
        
        self.mean: tuple = mean
        self.std: tuple = std
        self.augmentation: bool = augmentation
        self.randaugment: bool = randaugment

        transform = [
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std, inplace=False),
        ]

        if self.augmentation:
            transform = [
                T.RandomHorizontalFlip(0.5),
                T.RandAugment(num_ops=2, magnitude=9, ) if self.randaugment else IdentityTransform(),
            ] + transform

            transform += [
                CutoutOnFloat()
            ]

        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)


class PovertyMapTransform(nn.Module):
    _BAND_ORDER: list = [
        'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS'
    ]
    _MEANS_2009_17: dict = {
        'BLUE':  0.059183,
        'GREEN': 0.088619,
        'RED':   0.104145,
        'SWIR1': 0.246874,
        'SWIR2': 0.168728,
        'TEMP1': 299.078023,
        'NIR':   0.253074,
        'NIGHTLIGHTS': 5.101585,
        'DMSP':  4.005496,  # does not exist in current dataset
        'VIIRS': 1.096089,  # does not exist in current dataset
    }
    _STD_DEVS_2009_17: dict = {
        'BLUE':  0.022926,
        'GREEN': 0.031880,
        'RED':   0.051458,
        'SWIR1': 0.088857,
        'SWIR2': 0.083240,
        'TEMP1': 4.300303,
        'NIR':   0.058973,
        'NIGHTLIGHTS': 23.342916,
        'DMSP':  23.038301,  # does not exist in current dataset
        'VIIRS': 4.786354,   # does not exist in current dataset
    }
    size: tuple = (224, 224)

    def __init__(self, augmentation: bool = False, randaugment: bool = True, **kwargs):
        """All images have already been mean-subtracted and normalized."""
        super(PovertyMapTransform, self).__init__()
        
        self.mean: tuple = (self._MEANS_2009_17[k] for k in self._BAND_ORDER)    # XXX; unnces?
        self.std: tuple = (self._STD_DEVS_2009_17[k] for k in self._BAND_ORDER)  # XXX; unnecs?
        self.augmentation: bool = augmentation
        self.randaugment: bool = randaugment
        
        if self.augmentation:
            self.base_transform_on_multispectral = nn.Sequential(
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=0.1, scale=(0.9, 1.1)),
            )
            if self.randaugment:
                self.color_transform_on_rgb = nn.Sequential(
                    T.ConvertImageDtype(torch.uint8),
                    T.RandAugment(num_ops=2, magnitude=9, ),
                    T.ConvertImageDtype(torch.float),
                    T.ColorJitter(brightness=.8, contrast=.8, saturation=.8, hue=.1)
                )
            else:
                self.color_transform_on_rgb = T.ColorJitter(brightness=.8, contrast=.8, saturation=.8, hue=.1)
        else:
            self.base_transform_on_multispectral = None
            self.color_transform_on_rgb = None

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        
        # without augmentation
        if not self.augmentation:
            return x
        
        # with augmentation
        x = self.base_transform_on_multispectral(x)               # (N, 8, H, W)
        bgr_unnormalized = self._unnormalize_bgr(x[:, :3, :, :])  # (N, 3, H, W); in BGR order
        rgb_unnormalized = bgr_unnormalized[:, [2, 1, 0], :, :]   # (N, 3, H, W); in RGB order
        rgb_color_transformed = \
            self.color_transform_on_rgb(rgb_unnormalized)         # (N, 3, H, W); in RGB order
        bgr_color_transformed = \
            rgb_color_transformed[:, [2, 1, 0], :, :]             # (N, 3, H, W); in BGR order
        bgr_normalized = \
            self._normalize_bgr(bgr_color_transformed)            # (N, 3, H, W); in BGR order
        x[:, :3, :, :] = bgr_normalized
        x = self.cutout(x)
        
        return x

    def _unnormalize_bgr(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Unnormalize the first 3 channels in multispectral 8-channel image."""
        assert (x.ndim == 4) and (x.shape[1] == 3)
        return (x * self.bgr_stds.to(x.device)) + self.bgr_means.to(x.device)
    
    def _normalize_bgr(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert (x.ndim == 4) and (x.shape[1] == 3)
        return (x - self.bgr_means.to(x.device)) / self.bgr_stds.to(x.device)
        
    @property
    def bgr_means(self):
        return torch.tensor([self._MEANS_2009_17[c] for c in ['BLUE', 'GREEN', 'RED']]).view(-1, 1, 1)
    
    @property
    def bgr_stds(self):
        return torch.tensor([self._STD_DEVS_2009_17[c] for c in ['BLUE', 'GREEN', 'RED']]).view(-1, 1, 1)

    @staticmethod
    def cutout(img: torch.FloatTensor):
        
        assert (img.ndim == 4) and (img.shape[2] == img.shape[3])
        
        def _sample_uniform(a: float, b: float) -> float:
            return torch.empty(1).uniform_(a, b).item()
        
        _, _, _, width = img.shape
        cutout_width = _sample_uniform(0., width / 2.)
        cutout_center_x = _sample_uniform(0., width)
        cutout_center_y = _sample_uniform(0., width)
        
        x0 = int(max(0, cutout_center_x - cutout_width / 2))
        y0 = int(max(0, cutout_center_y - cutout_width / 2))
        x1 = int(min(width, cutout_center_x + cutout_width / 2))
        y1 = int(min(width, cutout_center_y + cutout_width / 2))

        # fill with zeros (expects zero-mean-normalized input for x)
        img[:, :, x0:x1, y0:y1] = 0.
        return img


class IWildCamTransform(nn.Module):
    size: typing.Tuple[int] = (448, 448)
    def __init__(self,
                 mean: tuple = (0.485, 0.456, 0.406),  # TODO; compute
                 std: tuple = (0.229, 0.224, 0.225),   # TODO; compute
                 augmentation: bool = False,
                 randaugment: bool = False,
                 **kwargs):
        super(IWildCamTransform, self).__init__()
        
        self.mean: tuple = mean
        self.std: tuple = std
        self.augmentation: bool = augmentation
        self.randaugment: bool = randaugment

        if self.augmentation:
            transform: list = [
                T.Resize(self.size),
                T.RandomHorizontalFlip(.5),
                T.RandomVerticalFlip(.5),
                T.RandAugment(num_ops=2, magnitude=9, ) if self.randaugment else IdentityTransform(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std, inplace=False),
                CutoutOnFloat(),
            ]
        else:
            transform: list = [
                T.Resize(self.size),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std, inplace=False),
            ]

        self.transform = nn.Sequential(*transform)
    
    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)


class RxRx1Transform(nn.Module):
    # TODO: use image-wise normalization?
    # TODO: add horizontal flip & random rotation to train transform
    means: dict = {
        'train': (0.027, 0.059, 0.042),
        'id_val': (0.027, 0.058, 0.042),
        'ood_val': (0.020, 0.095, 0.051),
        'test': (0.016, 0.060, 0.033),
    }
    stds: dict = {
        'train': (0.057, 0.057, 0.042),
        'id_val': (0.058, 0.056, 0.043),
        'ood_val': (0.031, 0.108, 0.038),
        'test': (0.021, 0.054, 0.024),
    }
    size: tuple = (256, 256)
    def __init__(self,
                 mean: tuple = (0.023, 0.062, 0.040),  # using statistics from full dataset;
                 std: tuple = (0.049, 0.062, 0.038),   # using statistics from full dataset
                 augmentation: bool = False,
                 randaugment: bool = True,
                 samplewise_normalize: bool = True,
                 **kwargs):
        super(RxRx1Transform, self).__init__()

        self.mean: tuple = mean  # TODO; try ImageNet means; [0.485, 0.456, 0.406]
        self.std: tuple = std    # TODO: try ImageNet stds;  [0.229, 0.224, 0.225]
        self.augmentation: bool = augmentation
        self.randaugment: bool = randaugment
        self.samplewise_normalize: bool = samplewise_normalize

        transform: list = [T.ConvertImageDtype(torch.float)]
        if samplewise_normalize:
            transform += [SamplewiseNormalize()]
        else:
            transform += [T.Normalize(self.mean, self.std, inplace=False)]

        if self.augmentation:
            transform = [
                RandomRotation90(),
                T.RandomHorizontalFlip(0.5),
                T.RandAugment(num_ops=2, magnitude=9, ) if self.randaugment else IdentityTransform(),
            ] + transform
        
        self.transform = nn.Sequential(*transform)
    
    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)


class CelebATransform(nn.Module):
    _original_resolution = (178, 218)
    def __init__(self,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225),
                 target_resolution: tuple = None,
                 augmentation: bool = False):
        super(CelebATransform, self).__init__()

        self.mean = mean
        self.std = std
        self.target_resolution = target_resolution
        self.augmentation = augmentation

        transform = [
            T.ConvertImageDtype(torch.float),
            T.Normalize(self.mean, self.std, inplace=False),
            T.CenterCrop(size=min(self._original_resolution)),
        ]

        if self.target_resolution is not None:
            transform += [T.Resize(size=self.target_resolution)]

        if self.augmentation:
            raise NotImplementedError
        
        self.transform = nn.Sequential(*transform)

    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)


class DomainBedTransform(nn.Module):
    _target_resolution = (224, 224)
    def __init__(self,
                 mean: tuple = (0.485, 0.456, 0.406),
                 std: tuple = (0.229, 0.224, 0.225),
                 augmentation: bool = False,
                 **kwargs):
        super(DomainBedTransform, self).__init__()

        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        
        if self.augmentation:
            _transform: list = [
                T.Resize(self._target_resolution),
                T.RandomResizedCrop(self._target_resolution[0], scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.RandomGrayscale(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std),
            ]
        else:
            _transform: list = [
                T.Resize(self._target_resolution),
                T.ConvertImageDtype(torch.float),
                T.Normalize(self.mean, self.std),
            ]

        self.transform = nn.Sequential(*_transform)

    def forward(self, x: torch.ByteTensor) -> torch.FloatTensor:
        return self.transform(x)



InputTransforms: typing.Dict[str, nn.Module] = {
    'camelyon17': Camelyon17Transform,
    'iwildcam': IWildCamTransform,
    'rxrx1': RxRx1Transform,
    'poverty': PovertyMapTransform,
    
    # 'camelyon17_ece': Camelyon17Transform,
    # 'poverty_ece': PovertyMapTransform,
    # 'rmnist': RotatedMNISTTransform,
    # 'cmnist': ColoredMNISTTransform,
    # 'pacs': DomainBedTransform,
    # 'vlcs': DomainBedTransform,
    # 'vlcs_ood': DomainBedTransform,
    # 'civilcomments': IdentityTransform,
    # 'celeba': CelebATransform,
}
