import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor, lambda_val):
        ctx.lambda_val = lambda_val
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val

    def set_lambda(self, lambda_val: float):
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim: int = 5000, hidden_dim: int = 500, output_dim: int = 100, p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return self.network(x)


class SentimentClassifier(nn.Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 100, output_dim: int = 2, p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 100, output_dim: int = 4, p: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)
