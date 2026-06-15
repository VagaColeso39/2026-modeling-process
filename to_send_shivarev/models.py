import torch.nn as nn
from config import *

class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_dim=SVD_COMPONENTS,
                 hidden_dims=GENERATOR_HIDDEN_DIMS, dropout_rate=DROPOUT_RATE):
        super().__init__()
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, z):
        return self.model(z)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Discriminator(nn.Module):
    def __init__(self, input_dim=SVD_COMPONENTS, hidden_dims=DISCRIMINATOR_HIDDEN_DIMS,
                 dropout_rate=DROPOUT_RATE):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)