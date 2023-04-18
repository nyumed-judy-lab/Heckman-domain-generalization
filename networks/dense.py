
import torch
import torch.nn as nn

default_activation = 'ELU'
default_dropout = 0.5
default_batchnorm = True
default_bias = True

class BasicNetwork(nn.Module):
    def __init__(self, layers,
                 activation=default_activation,
                 dropout=default_dropout,
                 batchnorm=default_batchnorm,
                 bias=default_bias):

        super(BasicNetwork, self).__init__()
        activation = getattr(nn, activation)

        _layers = []
        for d, (units_in, units_out) in enumerate(zip(layers, layers[1:])):
            _layers.append(nn.Linear(units_in, units_out, bias=bias))
            if d < len(layers) - 2:
                if batchnorm:
                    _layers.append(nn.BatchNorm1d(units_out))
                _layers.append(activation())
                if dropout:
                    _layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*_layers)

    def forward(self, x):
        return self.layers(x)

class HeckmanDNN(nn.Module):
    def __init__(self, f_layers, g_layers,
                 activation=default_activation,
                 dropout=default_dropout,
                 batchnorm=default_batchnorm,
                 bias=default_bias
                 ):

        super(HeckmanDNN, self).__init__()
        activation = getattr(nn, activation)

        _layers = []
        for d, (units_in, units_out) in enumerate(zip(f_layers, f_layers[1:])):
            _layers.append(nn.Linear(units_in, units_out, bias=bias))
            if d < len(f_layers) - 2:
                if batchnorm:
                    _layers.append(nn.BatchNorm1d(units_out))
                _layers.append(activation())
                if dropout:
                    _layers.append(nn.Dropout(dropout))

        self.f_layers = nn.Sequential(*_layers)

        _layers = []
        for d, (units_in, units_out) in enumerate(zip(g_layers, g_layers[1:])):
            _layers.append(nn.Linear(units_in, units_out, bias=bias))
            if d < len(g_layers) - 2:
                if batchnorm:
                    _layers.append(nn.BatchNorm1d(units_out))
                _layers.append(activation())
                if dropout:
                    _layers.append(nn.Dropout(dropout))

        self.g_layers = nn.Sequential(*_layers)

        self.rho = nn.Parameter(torch.zeros(g_layers[-1], dtype=torch.float), requires_grad=True)
        self.register_parameter(name='rho', param=self.rho)

    def forward(self, x):
        return torch.cat([self.f_layers(x), self.g_layers(x)], axis=1)

    def forward_f(self, x):
        return self.f_layers(x)

    def forward_g(self, x):
        return self.g_layers(x)

