# LSTM-Based Representation Learning for Momentum Trading

A deep learning approach to momentum trading using LSTM neural networks for stock price prediction and trading signal generation.

## Overview

This project implements an LSTM-based model to learn representations from historical stock price data and generate momentum trading signals. The model predicts future returns and creates long/short trading positions.

## Features

- Multi-stock portfolio support
- Feature engineering with technical indicators (log returns, volatility, volume metrics)
- LSTM architecture for sequential pattern recognition
- Automated trading signal generation
- Performance metrics (Sharpe ratio, returns, volatility)

## Installation

```bash
# Clone the repository
git clone https://github.com/helomelo1/LSTM-Based-Representation-Learning-for-Momentum-Trading.git
cd LSTM-Based-Representation-Learning-for-Momentum-Trading

# Install required packages
pip install torch numpy pandas yfinance
```

## Usage

```bash
python main.py
```

The script will download data, train the LSTM model, and output performance metrics.

## Model Architecture

- **Input**: Sequential features over a 20-day lookback window
- **Hidden Layer**: 64 LSTM units
- **Output**: Predicted return value
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Mean Squared Error

## Feature Set

1. Daily and 5-day log returns
2. 10-day volatility
3. Volume z-score
4. Typical price (average of high, low, close)

## Trading Strategy

- **Long Signal**: Predictions above 80th percentile
- **Short Signal**: Predictions below negative 80th percentile
- **Neutral**: Predictions between thresholds

## Performance Metrics

- Mean daily return
- Daily volatility
- Sharpe ratio (annualized)

## Dependencies

- PyTorch
- NumPy
- pandas
- yfinance
