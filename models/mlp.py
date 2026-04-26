import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, window_size, horizon):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(window_size, 32),
            nn.ReLU(),
            nn.Linear(32, horizon)
        )

    def forward(self, x):
        return self.model(x)