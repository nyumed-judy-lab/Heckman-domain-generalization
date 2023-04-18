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
