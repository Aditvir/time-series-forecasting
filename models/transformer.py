import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, hidden_size, horizon):
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])