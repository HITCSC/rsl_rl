import torch
import torch.nn as nn

class SmallCNNEncoder(nn.Module):
    def __init__(
            self, 
            embedding_dim=128,
    ):
        super(SmallCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 13, embedding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.cnn(x)


class LargeCNNEncoder(nn.Module):
    def __init__(
            self, 
            embedding_dim=512,
    ):
        super(LargeCNNEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=9, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=9, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)), 
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, embedding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.cnn(x)

