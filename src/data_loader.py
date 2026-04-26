import numpy as np
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    data = df.iloc[:, 1].values
    return data

def create_sequences(data, window_size, horizon):
    X, y = [], []

    for i in range(len(data) - window_size - horizon):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+horizon])

    return np.array(X), np.array(y)