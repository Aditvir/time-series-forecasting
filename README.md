# Time Series Forecasting Project

## Roll Number: 102317150

### Parameters:
- window_size = 8
- prediction_horizon = 3
- hidden_size = 14

## Models Implemented:
- MLP (baseline)
- RNN (assigned model)
- LSTM
- Transformer

## Methodology:
- Sliding window used to convert time series into supervised data
- Chronological train-test split (no shuffling)
- Models trained to predict next 3 values from past 8 values

## Evaluation Metrics:
- MSE
- MAE
- RMSE

## Observations:
- MLP fails due to lack of temporal understanding
- RNN captures sequence but struggles with long-term dependencies
- LSTM improves performance using memory gates
- Transformer captures global dependencies effectively

## Ablation Study:
- Smaller window size → insufficient context
- Larger window size → noisy and harder to train
- Optimal performance at moderate window size

## Dataset:
Download from:
https://www.kaggle.com/code/nageshsingh/predict-electricity-consumption

Place in:
data/Electric_Production.csv
