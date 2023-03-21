
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

from torch.utils.data import ConcatDataset, DataLoader
from networks.cnn_initializers import NetworkInitializer


class SeparatedHeckmanNetworkCNN(nn.Module):
    def __init__(self, args):
        self.args = args

        super(SeparatedHeckmanNetworkCNN, self).__init__()
        self.g_layers_backbone = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
        # 1-2. \omega_{g,k}, \forall{k} \in {1,...,K}
        self.g_layers_head = nn.Linear(
            in_features=self.g_layers_backbone.out_features, out_features=self.args.num_domains,
        )
        g_layers_ = list(self.g_layers_backbone.children())
        g_layers_.append(self.g_layers_head)
        self.g_layers = nn.Sequential(*g_layers_)

        # Outcome model: backbone
        self.f_layers_backbone = NetworkInitializer.initialize_backbone(
                name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
        # Outcome model: head
        self.f_layers_head = nn.Linear(
            in_features=self.f_layers_backbone.out_features,
            out_features=NetworkInitializer._out_features[self.args.data][0],
        )
        f_layers_ = list(self.f_layers_backbone.children())
        f_layers_.append(self.f_layers_head)
        self.f_layers = nn.Sequential(*f_layers_)
        
        self.rho = nn.Parameter(torch.zeros(args.num_domains, dtype=torch.float), requires_grad=True)
        self.register_parameter(name='rho', param=self.rho)

    def forward_f(self, x):
        return self.f_layers(x)

    def forward_g(self, x):
        return self.g_layers(x)

    def forward(self, x):
        return torch.cat([self.f_layers(x), self.g_layers(x)], axis=1)

    # def forward_f(self, x):
    #     x = self.f_layers_backbone(x)
    #     x = torch.cat([x, self.f_layers_head(x)], dim=1)
    #     return x #self.f_layers(x)

    # def forward_g(self, x):
    #     x = self.g_layers_backbone(x)
    #     x = torch.cat([x, self.g_layers_head(x)], dim=1)
    #     return x #self.g_layers(x)
    
##### TODO
# '''

# # 4-1. Correlation
# if self.defaults.loss_type == 'multiclass':
#     J: int = NetworkInitializer._out_features[self.args.data][0]  # number of outcome classes
#     K: int = len(self.train_domains)                              # number of training domains
#     _rho = torch.randn(K, J + 1, device=self.device, requires_grad=True)
#     self._rho = nn.Parameter(_rho)
# else:
#     K: int = len(self.train_domains)
#     _rho = torch.zeros(K, device=self.device, requires_grad=True)
#     self._rho = nn.Parameter(data=_rho)

# # 4-2. Sigma (for regression only)
# if self.defaults.loss_type == 'regression':
#     self.sigma = nn.Parameter(torch.ones(1, device=self.device), requires_grad=True)

# # 5. temperature
# self.temperature: typing.Union[torch.FloatTensor, float] = 1.0

# '''

# '''
# _layers = []
# for d, (units_in, units_out) in enumerate(zip(f_layers, f_layers[1:])):
#     _layers.append(nn.Linear(units_in, units_out, bias=bias))
#     if d < len(f_layers) - 2:
#         if batchnorm:
#             _layers.append(nn.BatchNorm1d(units_out))
#         _layers.append(activation())
#         if dropout:
#             _layers.append(nn.Dropout(dropout))

# self.f_layers = nn.Sequential(*_layers)

# _layers = []
# for d, (units_in, units_out) in enumerate(zip(g_layers, g_layers[1:])):
#     _layers.append(nn.Linear(units_in, units_out, bias=bias))
#     if d < len(g_layers) - 2:
#         if batchnorm:
#             _layers.append(nn.BatchNorm1d(units_out))
#         _layers.append(activation())
#         if dropout:
#             _layers.append(nn.Dropout(dropout))

# self.g_layers = nn.Sequential(*_layers)
# '''
