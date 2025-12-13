# LSTM-Based Representation Learning for Momentum Trading

A deep learning approach to momentum trading using LSTM (Long Short-Term Memory) neural networks for stock price prediction and trading signal generation on Indian stock market data.

## Overview

This project implements an LSTM-based model to learn representations from historical stock price data and generate momentum trading signals. The model predicts future returns and uses those predictions to create long/short trading positions for a portfolio of Indian stocks.

## Features

- **Multi-Stock Support**: Trades multiple stocks simultaneously (RELIANCE, TCS, INFOSYS, HDFC Bank, ICICI Bank)
- **Feature Engineering**: Includes various technical indicators:
  - 1-day and 5-day log returns (momentum)
  - 10-day volatility
  - Volume z-score
  - VWAP (Volume Weighted Average Price) deviation
- **LSTM Architecture**: Deep learning model for sequential pattern recognition
- **Automated Trading Signals**: Generates long/short signals based on predicted returns
- **Performance Metrics**: Sharpe ratio, mean return, and volatility calculations

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- yfinance
- pandas
- numpy

### Setup

```bash
# Clone the repository
git clone https://github.com/helomelo1/LSTM-Based-Representation-Learning-for-Momentum-Trading.git
cd LSTM-Based-Representation-Learning-for-Momentum-Trading

# Install required packages
pip install torch numpy pandas yfinance
```

## Usage

Simply run the main script:

```bash
python main.py
```

The script will:
1. Download historical stock data from Yahoo Finance
2. Engineer features from raw price data
3. Create sequential datasets for LSTM training
4. Train the LSTM model
5. Generate trading signals on test data
6. Output performance metrics

## Model Architecture

### LSTM Model
- **Input**: Sequential features over a 20-day lookback window
- **Hidden Layer**: 64 LSTM units
- **Output**: Single regression value (predicted return)

### Training Configuration
- **Lookback Period**: 20 days
- **Batch Size**: 64
- **Epochs**: 10
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)

## Data Pipeline

### Stocks Traded
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- ICICIBANK.NS (ICICI Bank)

### Data Period
- **Start Date**: 2016-01-01
- **End Date**: 2024-01-01
- **Interval**: Daily (1d)

### Feature Set
1. **log_returns_1d**: Daily log returns
2. **log_returns_5d**: 5-day rolling momentum
3. **vol_10d**: 10-day volatility
4. **vol_z**: Volume z-score (normalized volume)
5. **vwap**: Volume Weighted Average Price (average of high, low, close)

## Trading Strategy

The model generates trading signals based on predicted returns:
- **Long Signal**: Predictions above the 80th percentile threshold
- **Short Signal**: Predictions below negative 80th percentile threshold
- **Neutral**: Predictions between these thresholds

## Performance Metrics

The system outputs:
- **Mean Daily Return**: Average daily return of the strategy
- **Daily Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return metric (annualized)

## Project Structure

```
.
├── main.py          # Main training and evaluation script
├── data.py          # Data loading and feature engineering
└── README.md        # Project documentation
```

## Dependencies

- `torch`: PyTorch for deep learning
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `yfinance`: Yahoo Finance data download

## Technical Details

### Data Processing
- Historical stock data is downloaded via yfinance API
- Technical features are engineered including momentum, volatility, and volume indicators
- Data is structured into sequences for LSTM input

### Model Training
- 80/20 train-test split
- Batch training with shuffling on training set
- Sequential evaluation on test set

### Signal Generation
- Predictions are ranked using percentiles
- Top predictions receive long signals (+1)
- Bottom predictions receive short signals (-1)
- Strategy returns are calculated as signal × actual return

## Disclaimer

This project is for educational and research purposes only. It is not financial advice. Trading stocks involves risk, and past performance does not guarantee future results.

## License

This project is open source and available for educational purposes.

## Author

helomelo1
