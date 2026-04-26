import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from src.data_loader import load_data, create_sequences
from src.utils import evaluate

from models.mlp import MLP
from models.rnn import RNNModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel


# Parameters
window_size = 16
prediction_horizon = 3
hidden_size = 14


# Load Data
data = load_data("data/Electric_Production.csv")

X, y = create_sequences(data, window_size, prediction_horizon)

# Split
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


# Normalization
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train.reshape(-1,1)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)


# Train Function
def train(model, X_train, y_train):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []

    for epoch in range(20):
        model.train()

        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32)

        if isinstance(model, MLP):
            outputs = model(inputs)
        else:
            inputs = inputs.unsqueeze(-1)
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Epoch {epoch+1}: {loss.item()}")

    return losses


# Evaluation Function
def test(model, X_test, y_test):
    model.eval()

    inputs = torch.tensor(X_test, dtype=torch.float32)

    if isinstance(model, MLP):
        preds = model(inputs).detach().numpy()
    else:
        preds = model(inputs.unsqueeze(-1)).detach().numpy()

    mse, mae, rmse = evaluate(y_test, preds)

    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

    return preds

# Run Models

models = {
    "MLP": MLP(window_size, prediction_horizon),
    "RNN": RNNModel(hidden_size, prediction_horizon),
    "LSTM": LSTMModel(hidden_size, prediction_horizon),
    "Transformer": TransformerModel(hidden_size, prediction_horizon)
}

for name, model in models.items():
    print(f"\nTraining {name}...\n")

    losses = train(model, X_train, y_train)
    preds = test(model, X_test, y_test)

    # Plot Loss
    plt.plot(losses)
    plt.title(f"{name} Training Loss")
    plt.show()

    # Plot Predictions
    plt.plot(y_test[:50], label="Actual")
    plt.plot(preds[:50], label="Predicted")
    plt.title(f"{name} Predictions")
    plt.legend()
    plt.show()