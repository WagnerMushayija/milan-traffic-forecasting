import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load data
FULL_PATH = 'milan_full_sorted.parquet'
df = pd.read_parquet(FULL_PATH)
df['time_interval']=pd.to_datetime(df['time_interval'])

# Target areas and test window
TARGET_AREAS = [6260, 4159, 4556]
test_start = '2013-12-16'
test_end = '2013-12-22'

# resample to hourly

def resample_to_hourly(df, value_col='internet_traffic', time_col='time_interval', id_col='square_id'):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    hourly = (
        df.groupby([id_col, pd.Grouper(key=time_col, freq='h')])[value_col]
          .mean()
          .reset_index()
    )
    return hourly

print('Resampling to hourly...')
df_hourly = resample_to_hourly(df)

test_start_h = pd.Timestamp(test_start).floor('h')
test_end_h = pd.Timestamp(test_end).floor('h')
print('df_hourly shape', df_hourly.shape)

# helper

def make_sequences(series, lookback):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback:i])
        y.append(series[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


def train_lstm(model, X_train, y_train, epochs, batch_size, patience=5):
    dataset   = TensorDataset(
        torch.tensor(X_train).unsqueeze(-1),
        torch.tensor(y_train)
    )
    val_size  = max(1, int(0.1 * len(dataset)))
    train_size= len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion  = nn.L1Loss()

    best_val   = float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.float(), yb.float()
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.float(), yb.float()
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def calculate_metrics(actual, predicted):
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return mae, rmse, mape

# experiments
lstm_experiments = {
    'Exp1_Baseline': {
        'lookback':    24,
        'hidden_size': 64,
        'num_layers':  2,
        'dropout':     0.0,
        'epochs':      10,
        'batch_size':  32,
    },
    'Exp2_Dropout': {
        'lookback':    24,
        'hidden_size': 64,
        'num_layers':  2,
        'dropout':     0.2,
        'epochs':      10,
        'batch_size':  32,
    },
}

device = torch.device('cpu')

final_lstm = {}
lstm_metrics = {}

for area in TARGET_AREAS:
    print('\n=== AREA', area, '===')
    area_df = df_hourly[df_hourly['square_id']==area].set_index('time_interval')
    n_train = (area_df.index < test_start_h).sum()
    n_test = ((area_df.index >= test_start_h)&(area_df.index<=test_end_h)).sum()
    print('n_train, n_test', n_train, n_test)
    if n_train==0 or n_test==0:
        print('skip area')
        continue

    series = area_df['internet_traffic'].values.reshape(-1,1).astype(np.float32)
    scaler = MinMaxScaler(); scaler.fit(series[:n_train])
    series_norm = scaler.transform(series).flatten()

    best_mae = float('inf')
    best_pred = None
    area_results = {}

    for name,p in lstm_experiments.items():
        print(' running', name)
        lookback = p['lookback']
        X_all, y_all = make_sequences(series_norm, lookback)
        n_train_seq = n_train - lookback
        if n_train_seq <= 0:
            print(' not enough train sequences for lookback')
            continue
        X_train = X_all[:n_train_seq]; y_train = y_all[:n_train_seq]
        X_test = X_all[n_train_seq:n_train_seq + n_test]; y_test_norm = y_all[n_train_seq:n_train_seq + n_test]
        y_test_real = scaler.inverse_transform(y_test_norm.reshape(-1,1)).flatten()

        model = LSTMModel(hidden_size=p['hidden_size'], num_layers=p['num_layers'], dropout=p['dropout']).to(device)
        model = train_lstm(model, X_train, y_train, epochs=p['epochs'], batch_size=p['batch_size'], patience=3)

        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test).unsqueeze(-1).float()
            pred_norm = model(X_test_t).cpu().numpy()
        pred_real = scaler.inverse_transform(pred_norm.reshape(-1,1)).flatten()

        mae, rmse, mape = calculate_metrics(y_test_real, pred_real)
        print('  result:', mae, rmse, mape)
        area_results[name] = {'mae':mae,'rmse':rmse,'mape':mape,'pred':pred_real}
        if mae < best_mae:
            best_mae = mae; best_pred = pred_real

    final_lstm[area]=best_pred
    lstm_metrics[area]=area_results

print('\n=== SUMMARY ===')
for area,results in lstm_metrics.items():
    print('AREA', area)
    for name,r in results.items():
        print(' ',name, r['mae'], r['mape'])

