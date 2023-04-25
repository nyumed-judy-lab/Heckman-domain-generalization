import torch
import torch.nn as nn
import torch.nn.functional as F
import typing
from torch.utils.data import ConcatDataset, DataLoader
from networks.cnn_initializers import NetworkInitializer

class HeckmanCNN(nn.Module):
    def __init__(self, args):
        self.args = args
        super(HeckmanCNN, self).__init__()

        # Selection model: backbone
        self.g_layers_backbone = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
        # Selection model: Head
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
            out_features=NetworkInitializer._out_features[args.data][0],
        )
        f_layers_ = list(self.f_layers_backbone.children())
        f_layers_.append(self.f_layers_head)
        self.f_layers = nn.Sequential(*f_layers_)
        
        if self.args.loss_type == 'multiclass':
            # rho (Multiclass Classification)
            J: int = NetworkInitializer._out_features[self.args.data][0]  # number of outcome classes
            K: int = len(self.args.train_domains)                         # number of training domains
            self.rho = nn.Parameter(torch.randn(K, J + 1, dtype=torch.float), requires_grad=True)
            # self.rho = nn.Parameter(torch.rand(y_true.shape[0], y_true_.shape[1]+1, dtype=torch.float), requires_grad=True)
            # J: int = NetworkInitializer._out_features[args.data][0]  # number of outcome classes
            # K: int = len(args.train_domains)                         # number of training domains
            # rho = nn.Parameter(torch.randn(K, J + 1, dtype=torch.float), requires_grad=True)
            
        else:
            # rho (Binary Classification)
            K: int = len(args.train_domains)
            self.rho = nn.Parameter(torch.zeros(K, dtype=torch.float), requires_grad=True)
        # self.register_parameter(name='rho', param=self.rho)
        # Sigma (regression only if )
        if self.args.loss_type == 'regression':
            self.sigma = nn.Parameter(torch.ones(1, device=self.args.device), requires_grad=True)
        
        self.temperature: typing.Union[torch.FloatTensor, float] = 1.0

    def forward_f(self, x):
        return self.f_layers(x)

    def forward_g(self, x):
        return self.g_layers(x)

    def forward(self, x):
        return torch.cat([self.f_layers(x), self.g_layers(x)], axis=1)



class HeckmanCNN_BinaryClass(nn.Module):
    def __init__(self, args):
        self.args = args

        super(HeckmanCNN_BinaryClass, self).__init__()
        self.g_layers_backbone = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
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

class HeckmanCNN_MultiClass(nn.Module):
    def __init__(self, args):
        self.args = args

        super(HeckmanCNN_MultiClass, self).__init__()
        self.g_layers_backbone = NetworkInitializer.initialize_backbone(
            name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
        )
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
        
        J: int = NetworkInitializer._out_features[self.args.data][0]  # number of outcome classes
        K: int = len(self.args.train_domains)                         # number of training domains
        self.rho = nn.Parameter(torch.randn(K, J + 1, dtype=torch.float), requires_grad=True)
        # self.rho = nn.Parameter(torch.rand(y_true.shape[0], y_true_.shape[1]+1, dtype=torch.float), requires_grad=True)
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

# class HeckmanCNN(nn.Module):
#     def __init__(self, args):
#         self.args = args

#         super(HeckmanCNN, self).__init__()
#         self.g_layers_backbone = NetworkInitializer.initialize_backbone(
#             name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
#         )
#         # 1-2. \omega_{g,k}, \forall{k} \in {1,...,K}
#         self.g_layers_head = nn.Linear(
#             in_features=self.g_layers_backbone.out_features, out_features=self.args.num_domains,
#         )
#         g_layers_ = list(self.g_layers_backbone.children())
#         g_layers_.append(self.g_layers_head)
#         self.g_layers = nn.Sequential(*g_layers_)

#         # Outcome model: backbone
#         self.f_layers_backbone = NetworkInitializer.initialize_backbone(
#                 name=self.args.backbone, data=self.args.data, pretrained=self.args.pretrained,
#         )
#         # Outcome model: head
#         self.f_layers_head = nn.Linear(
#             in_features=self.f_layers_backbone.out_features,
#             out_features=NetworkInitializer._out_features[self.args.data][0],
#         )
#         f_layers_ = list(self.f_layers_backbone.children())
#         f_layers_.append(self.f_layers_head)
#         self.f_layers = nn.Sequential(*f_layers_)
        
#         self.rho = nn.Parameter(torch.zeros(args.num_domains, dtype=torch.float), requires_grad=True)
#         self.register_parameter(name='rho', param=self.rho)

#     def forward_f(self, x):
#         return self.f_layers(x)

#     def forward_g(self, x):
#         return self.g_layers(x)

#     def forward(self, x):
#         return torch.cat([self.f_layers(x), self.g_layers(x)], axis=1)

    
