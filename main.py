import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data import load_dataset

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META",
    "NVDA","JPM","BAC","XOM","CVX",
    "JNJ","PFE","KO","PEP","WMT",
    "DIS","NFLX","INTC","AMD","TSLA",
    "CAT","BA","GE","MMM","IBM",
    "GS","MS","AXP","V","MA"
]
START_DATE = "2016-01-01"
END_DATE = "2024-01-01"
INTERVAL = "1d"
LOOKBACK = 20
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTM_model(nn.Module):
    def __init__(self, num_features, hidden_size=64):#, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            # num_layers=num_layers,
            # dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        # h = self.dropout(h)
        out = self.fc(h).squeeze(-1)

        return out
    

X, y = load_dataset(
    ticker=TICKERS,
    start=START_DATE,
    end=END_DATE,
    interval=INTERVAL,
    lookback=LOOKBACK,
)

print(f"Data Shape: {X.shape}, {y.shape}")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = LSTM_model(num_features=X.shape[2]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {total_loss:.4f}")


model.eval()
preds = []

with torch.no_grad():
    for xb, _ in test_loader:
        xb = xb.to(DEVICE)
        preds.append(model(xb).cpu())

preds = torch.cat(preds).numpy()
true = y_test.numpy()

threshold = np.percentile(preds, 80)

signals = np.zeros_like(preds)
signals[preds > threshold] = 1  # Long Signal
signals[preds < -threshold] = -1   # Short Signal

strategy_returns = signals * true
mean_return = strategy_returns.mean()
std_return = strategy_returns.std()

sharpe = mean_return / std_return * np.sqrt(252)


print(f"Mean daily return: {mean_return:.6f}")
print(f"Daily volatility : {std_return:.6f}")
print(f"Sharpe ratio     : {sharpe:.2f}")