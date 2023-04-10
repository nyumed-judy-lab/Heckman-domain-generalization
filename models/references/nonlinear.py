
import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50, resnet101, densenet121


"""
    Remove this file.
"""



TORCHVISION_MODELS = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'densenet121': densenet121
}

DATA_INPUT_CHANNELS = {
    'rmnist': 1,
    'cmnist': 2,
    'camelyon': 3,
    'rxrx1': 3,
    'poverty': 8,
}


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


class ResNetBackbone(ConvBackboneBase):
    def __init__(self,
                 name: str = 'resnet50',
                 data: str = None,
                 pretrained: bool = False):
        super(ResNetBackbone, self).__init__()
    
        self.name: str = name
        self.data: str = data
        self.pretrained: bool = pretrained

        self.in_channels: int = DATA_INPUT_CHANNELS[self.data]
        if self.in_channels != 3:
            raise NotImplementedError  # TODO

        self.layers = TORCHVISION_MODELS[self.name](pretrained=self.pretrained)
        self.layers = self.remove_gap_and_fc(self.layers)
        
        if not self.pretrained:
            initialize_weights(self.layers, activation='relu')

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)
    
    @staticmethod
    def remove_gap_and_fc(resnet: nn.Module) -> nn.Module:
        model = nn.Sequential()
        for name, child in resnet.named_children():
            if name not in ['avgpool', 'fc']:
                model.add_module(name, child)  # preserve original names
        
        return model

    @property
    def out_channels(self):
        if self.name == 'resnet18':
            return 512
        elif self.name == 'resnet50':
            return 2048
        elif self.name == 'resnet101':
            raise NotImplementedError
        else:
            raise KeyError


def initialize_weights(model: nn.Module, activation: str = 'relu'):
    """Initialize trainable weights."""
    
    for _, m in model.named_modules():
        # Convolution layers
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # Batch normalization layers
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass
        # Linear layers
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass


class SimpleLeNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super(SimpleLeNetBackbone, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=5, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        initialize_weights(self)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)


class SimpleLeNet(nn.Module):  # XXX: deprecated, remove.
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(SimpleLeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(16, num_classes)

        initialize_weights(self)

    def forward(self, x: torch.FloatTensor):
        x = self.gap(self.conv(x))
        return self.linear(x.squeeze())
