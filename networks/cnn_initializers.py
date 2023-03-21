
import typing
import collections

import torch
import torch.nn as nn

# from networks.cnn_backbone import ResNetBackbone
# from networks.cnn_backbone import DenseNetBackbone
# from networks.cnn import CNNForMNIST
# from networks.bert import DistilBertBackbone

class NetworkInitializer(object):
    
    _supported_datasets: typing.List[str] = [
        'rmnist',
        'cmnist',
        'pacs',
        'vlcs',
        'vlcs_ood',
        'camelyon17',
        'camelyon17_ece',
        'rxrx1',
        'poverty',
        'poverty_ece',
        'celeba',
        'insight',
        'civilcomments',
        'iwildcam',
        'fmow',
    ]

    _in_channels: typing.Dict[str, int] = {
        'rmnist': 1,
        'cmnist': 3,
        'pacs': 3, #####
        'vlcs': 3, #####
        'vlcs_ood': 3, #####
        'camelyon17': 3, #####
        'camelyon17_ece': 3, #####
        'rxrx1': 3,
        'poverty': 8,
        'poverty_ece': 8,
        'celeba': 3,
        'insight': 1,
        'iwildcam': 3,
        'fmow': 3,
    }

    _out_features: typing.Dict[str, typing.Tuple[int, str]] = {
        'rmnist': (10, 'multiclass'),
        'cmnist': (1, 'binary'),
        'pacs': (7, 'multiclass'), #####
        'vlcs': (5, 'multiclass'), #####
        'vlcs_ood': (5, 'multiclass'), #####
        'camelyon17': (1, 'binary'),
        'camelyon17_ece': (1, 'binary'),
        'rxrx1': (1139, 'multiclass'),
        'poverty': (1, 'regression'),
        'poverty_ece': (1, 'regression'),
        'celeba': (1, 'binary'),
        'insight': (1, 'binary'),
        'civilcomments': (1, 'binary'),
        'iwildcam': (182, 'multiclass'),
        'fmow': (62, 'multiclass'),
    }
    
    @classmethod
    def initialize_backbone(cls, name: str, data: str, pretrained: bool) -> nn.Module:
        """
        Helper function for initializing backbones (encoders).
        """
        if data not in cls._supported_datasets:
            raise ValueError(
                f"Invalid option for data (={data}). Supports {', '.join(cls._supported_datasets)}."
            )
        if data in ['civilcomments', 'amazon', ]:
            return cls.initialize_bert_backbone(name=name, data=data, pretrained=pretrained)
        elif data in ['cmnist', 'rmnist', ]:
            return CNNForMNIST(in_channels=cls._in_channels[data])
        else:
            return cls.initialize_cnn_backbone(name=name, data=data, pretrained=pretrained)

    @classmethod
    def initialize_cnn_backbone(cls, name: str, data: str, pretrained: bool) -> nn.Module:
        """
        Helper function for initializing CNN-based backbones.
            nn.Sequential(backbone, global_average_pooling, flatten)
        """
        in_channels: int = cls._in_channels[data]
        if name.startswith('resnet'):
            return ResNetBackbone(name=name, in_channels=in_channels, pretrained=pretrained)
        elif name.startswith('wideresnet'):
            raise NotImplementedError("Work in progress.")
        elif name.startswith('densenet'):
            return DenseNetBackbone(name=name, in_channels=in_channels, pretrained=pretrained)
        else:
            raise ValueError("Only supports {cnn, resnet, wideresnet, densenet}-based models.")

    @classmethod
    def initialize_bert_backbone(cls, name: str, **kwargs) -> nn.Module:
        if name not in ['distilbert-base-uncased', ]:
            raise NotImplementedError
        return DistilBertBackbone(name=name)

    @classmethod
    def initialize_linear_output_layer(cls, data: str, backbone: nn.Module, add_features: int = 0) -> nn.Module:
        """
        Helper function for initializing output {classification, regression} layer.
        Arguments:
            data: str,
            backbone: nn.Module,
            add_features: int,
        Returns:
            nn.Linear
        """

        out_features, _ = cls._out_features[data]
        if isinstance(backbone, (ResNetBackbone, DenseNetBackbone)):
            in_features: int = backbone.out_features + add_features
            linear = nn.Linear(in_features, out_features, bias=True)
            linear.bias.data.fill_(0.)
            return linear
        else:
            raise NotImplementedError(f"Backbone: `{backbone.__class__.__name__}`")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import densenet121, densenet161, densenet169, densenet201


class ConvBackboneBase(nn.Module):
    def __init__(self):
        super(ConvBackboneBase, self).__init__()
    
    def forward(self, x: torch.FloatTensor):
        raise NotImplementedError
    
    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str, key: str, device: str = 'cpu') -> None:
        ckpt = torch.load(path, map_location=device)
        self.load_state_dict(ckpt[key])
    
    def freeze_weights(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.view(batch_size, -1)

class DenseNetBackbone(ConvBackboneBase):
    def __init__(self,
                 name: str = 'densenet121',
                 in_channels: int = 3,
                 pretrained: bool = False):
        super(DenseNetBackbone, self).__init__()

        self.name: str = name
        self.in_channels: int = in_channels
        self.pretrained: bool = pretrained

        _densenet = self._build_with_torchvision()
        self.layers = self.keep_backbone_only(_densenet, gap_and_flatten=True)

        if self.in_channels != 3:
            self.layers = self.change_first_conv_input_channels(self.layers, c=self.in_channels)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        #return F.normalize(self.layers(x), p=2, dim=-1)
        return self.layers(x)

    def _build_with_torchvision(self):
        if self.name == 'densenet121':
            return densenet121(pretrained=self.pretrained)
        elif self.name == 'densnet161':
            return densenet161(pretrained=self.pretrained)
        elif self.name == 'densnet169':
            return densenet169(pretrained=self.pretrained)
        elif self.name == 'densnet201':
            return densenet201(pretrained=self.pretrained)
        else:
            raise NotImplementedError
    
    @staticmethod
    def keep_backbone_only(densenet: nn.Module, gap_and_flatten: bool = True) -> nn.Module:
        """Add function docstring."""
        model = nn.Sequential()
        for name, child in densenet.named_children():
            assert name in ['features', 'classifier']
            if name == 'features':
                model.add_module(name, child)
        # we refer readers to the forward function of
        # `torchvision.models.densenet.DenseNet`.
        model.add_module('relu', nn.ReLU(inplace=False))

        if gap_and_flatten:
            model.add_module('gap', nn.AdaptiveAvgPool2d(1))
            model.add_module('flatten', Flatten())
        
        return model
    
    @staticmethod
    def change_first_conv_input_channels(densenet: nn.Module, c: int) -> nn.Module:
        """Add function docstring."""
        model = nn.Sequential()
        for name, child in densenet.named_children():
            assert name in ['features', 'relu']
            if name == 'features':
                # Find the first conv layer and replace it
                sub_model = nn.Sequential()
                for sub_name, sub_child in child.named_children():
                    if sub_name == 'conv0':
                        assert hasattr(sub_child, 'out_channels')
                        first_conv = nn.Conv2d(in_channels=c, out_channels=sub_child.out_channels,
                                               kernel_size=7, stride=2, padding=3, bias=False)
                        nn.init.kaiming_normal_(first_conv.weight, mode='fan_in', nonlinearity='relu')
                        sub_model.add_module(sub_name, first_conv)  # first conv
                        raise NotImplementedError
                    else:
                        # All the other layers
                        sub_model.add_module(sub_name, sub_child)
                # Add backbone with first conv layer fixed
                model.add_module(name, sub_model)
            else:
                # ReLU layer from `keep_backbone_only`
                model.add_module(name, child)       

        return model      
        
    @property
    def out_channels(self) -> int:
        if self.name == 'densenet121':
            return 1024
        elif self.name == 'densenet161':
            return 2208
        elif self.name == 'densenet161':
            return 1664
        elif self.name == 'densenet201':
            return 1920
        else:
            raise NotImplementedError

    @property
    def out_features(self) -> int:
        return self.out_channels
    


class ResNetBackbone(ConvBackboneBase):
    def __init__(self,
                 name: str = 'resnet50',
                 in_channels: int = 3,
                 pretrained: bool = False):
        super(ResNetBackbone, self).__init__()

        self.name: str = name
        self.in_channels: int = in_channels
        self.pretrained: bool = pretrained

        _resnet = self._build_with_torchvision()
        self.layers = self.fetch_backbone_only(_resnet, gap_and_flatten=True)

        if self.in_channels != 3:
            """Change the input channels of the first convolution.
            In this case, restrict the use of pretrained models."""
            self.layers = self.change_first_conv_input_channels(self.layers, c=self.in_channels)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # return F.normalize(self.layers(x), p=2, dim=-1)  # TODO:
        return self.layers(x)

    def _build_with_torchvision(self):
        if self.name == 'resnet18':
            return resnet18(pretrained=self.pretrained)
        elif self.name == 'resnet50':
            return resnet50(pretrained=self.pretrained)
        elif self.name == 'resnet101':
            return resnet101(pretrained=self.pretrained)
        else:
            raise NotImplementedError

    @staticmethod
    def fetch_backbone_only(resnet: nn.Module, gap_and_flatten: bool = True) -> nn.Module:
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)
        
        if gap_and_flatten:
            model.add_module('gap', nn.AdaptiveAvgPool2d(1))
            model.add_module('flatten', Flatten())

        return model

    @staticmethod
    def remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        """
        Helper function which removes:
            1) global average pooling
            2) fully-connected head
        """
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)

        return model

    @staticmethod
    def change_first_conv_input_channels(resnet: nn.Module, c: int) -> nn.Module:
        """Add function docstring."""
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name == 'conv1':
                assert hasattr(child, 'out_channels')
                first_conv = nn.Conv2d(in_channels=c, out_channels=child.out_channels,
                                       kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(first_conv.weight, mode='fan_out', nonlinearity='relu')
                model.add_module(name, first_conv)
            else:
                model.add_module(name, child)
        return model

    @property
    def out_channels(self) -> int:
        if self.name == 'resnet18':
            return 512
        elif self.name == 'resnet50':
            return 2048
        elif self.name == 'resnet101':
            return 2048
        else:
            raise KeyError

    @property
    def out_features(self) -> int:
        return self.out_channels
