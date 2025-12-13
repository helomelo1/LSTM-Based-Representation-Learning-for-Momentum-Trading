import yfinance as yf
import pandas as pd
import numpy as np


def download_data(tickers, start, end, interval='1d') -> pd.DataFrame:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        progress=False
    )

    frames = []

    for ticker in tickers:
        df = data[ticker].copy()
        df['ticker'] = ticker
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]

        frames.append(df)

    df = pd.concat(frames, axis=0)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['log_returns_1d'] = np.log(df['adj_close']) - np.log(df.groupby('ticker')['adj_close'].shift(1)) # Log Return
    df['log_returns_5d'] = df.groupby('ticker')['log_returns_1d'].rolling(5).sum().reset_index(level=0, drop=True) # Rolling Momentum

    df['vol_10d'] = (
        df.groupby('ticker')['log_returns_1d'].rolling(10).std().reset_index(level=0, drop=True)
    )

    df['vol_z'] = (
        df.groupby('ticker')['volume'].transform(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())
    )

    df['vwap'] = (df['high'] +df['low'] + df['close']) / 3
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']

    return df


def make_seq(df: pd.DataFrame, feature_columns, lookback=20, horizon=5):
    X, y = [], []

    for ticker, group in df.groupby('ticker'):
        group = group.dropna().reset_index(drop=True)

        for i in range(lookback, len(group) - horizon):
            X.append(group.loc[i - lookback:i - 1][feature_columns].values)
            future_return = group.iloc[i : i + horizon]['log_returns_1d'].sum()
            y.append(future_return)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


def load_dataset(ticker, start, end, interval='1d', lookback=20):
    df = download_data(tickers=ticker, start=start, end=end, interval=interval)
    df = add_features(df)

    feature_cols = [
        'log_returns_1d',
        'log_returns_5d',
        'vol_10d',
        'vol_z',
        'vwap'
    ]

    X, y = make_seq(df, feature_columns=feature_cols, lookback=lookback)

    return X, y
