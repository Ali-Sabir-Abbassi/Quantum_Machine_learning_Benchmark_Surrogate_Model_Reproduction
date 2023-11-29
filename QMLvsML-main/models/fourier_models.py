import torch 
import torch.nn.functional as F
from torch.linalg import lstsq
import torch.nn as nn
import numpy as np 


class Fourier_model(nn.Module):
    def __init__(self, frequencies, output_dim=1, sin_weights=None, cos_weights=None):
        super(Fourier_model, self).__init__()  # Python 3 syntax
        self.W = frequencies.t().float()  # Ensure frequencies are float32
        self.input_dim = frequencies.shape[0]  # length of one frequency vector
        self.output_dim = output_dim
        # Initialize layers with .float() to ensure float32
        self.linear_sin = nn.Linear(self.input_dim, self.output_dim, bias=False).float()
        self.linear_cos = nn.Linear(self.input_dim, self.output_dim, bias=False).float()
        # Set weights if provided, ensuring they are the correct shape and type
        if sin_weights is not None:
            self.linear_sin.weight.data.copy_(sin_weights.float().view(self.linear_sin.weight.size()))
        if cos_weights is not None:
            self.linear_cos.weight.data.copy_(cos_weights.float().view(self.linear_cos.weight.size()))

    def forward(self, x):
        z = x.matmul(self.W)
        sin_z = self.linear_sin(z.sin())
        cos_z = self.linear_cos(z.cos())
        output = sin_z + cos_z
        return output
