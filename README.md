# Quantlens

A quantitative finance machine learning framework for predicting stock price movements using technical analysis and advanced ML models.

## 📊 Overview

Quantlens is a comprehensive platform that combines financial data engineering, feature engineering, and machine learning to predict 5-day stock returns. It includes:

- **Data Pipeline**: Automated fetching and processing of historical stock data from multiple markets
- **Feature Engineering**: 10+ technical indicators and statistical features (RSI, MACD, Stochastic Oscillator, SMA, momentum, volatility)
- **Model Training**: Ensemble of three ML models (Linear Regression, Random Forest, XGBoost)
- **Backtesting**: Performance evaluation against historical data
- **Interactive Dashboard**: Streamlit-based web app for real-time analysis and visualization

## 🚀 Features

### Data Support
- **S&P 500**: US large-cap equity universe
- **NIFTY 50**: Indian large-cap equity universe (extensible design)

### Technical Indicators
- RSI (Relative Strength Index) - 14 day window
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator (%K and %D)
- Simple Moving Averages (10-day, 50-day)
- Volatility (20-day rolling)
- Momentum (10-day)

### ML Models
1. **Linear Regression** - Baseline interpretability
2. **Random Forest** - Non-linear patterns (100 trees, max depth 5)
3. **XGBoost** - Advanced gradient boosting (300 trees, learning rate 0.05)

## 📁 Project Structure

```
quantlens/
├── app/
│   └── streamlit_app.py          # Interactive dashboard
├── src/
│   ├── datapipeline.py           # Data fetching and universe management
│   ├── features.py               # Technical indicator engineering
│   ├── models.py                 # Model training and evaluation
│   ├── backtest.py               # Backtesting framework
│   └── explainability.py         # Model interpretation tools
├── data/
│   ├── AAPL.csv                  # Sample stock data (Apple)
│   ├── GOOG.csv                  # Sample stock data (Google)
│   ├── MSFT.csv                  # Sample stock data (Microsoft)
│   ├── processed_features.csv    # Engineered features dataset
│   └── notebooks/                # Jupyter notebooks for exploration
│       ├── data_fetching.ipynb
│       ├── feature_engineering.ipynb
│       ├── model_training.ipynb
│       └── backtest_and_analysis.ipynb
└── pyproject.toml                # Project configuration
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantlens.git
cd Quantlens
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## 📖 Usage

### Quick Start: Interactive Dashboard

Run the Streamlit app for an interactive experience:
```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Real-time stock data visualization
- Technical indicator plots
- Model predictions and confidence intervals
- Backtesting performance metrics
- Feature importance analysis

### Python API

#### 1. Fetch Data and Generate Features

```python
from src.datapipeline import get_sp500_tickers, TICKERS
from src.features import generate_features
import yfinance as yf

# Fetch data
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")

# Generate features
features_df = generate_features(data)
print(features_df.head())
```

#### 2. Train Models

```python
from src.models import train_models, evaluate_models
from sklearn.model_selection import train_test_split

# Split data
X = features_df.drop(['Close', 'target_5d'], axis=1)
y = features_df['target_5d']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
models = train_models(X_train, y_train)

# Evaluate
evaluate_models(models, X_test, y_test)
```

#### 3. Backtest Strategy

```python
from src.backtest import backtest_strategy

results = backtest_strategy(
    models=models,
    features_df=features_df,
    initial_capital=100000,
    position_size=0.05  # Risk 5% per trade
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 📊 Data Format

Input data must contain OHLCV columns:
- **Date**: Trading date (index)
- **Open**: Opening price
- **High**: Daily high
- **Low**: Daily low
- **Close**: Closing price
- **Volume**: Trading volume

Target variable: **target_5d** (5-day forward return)

## 📈 Model Performance

Models are evaluated on unseen test data using:
- **R² Score**: Measures variance explained (0 to 1, higher is better)
- **RMSE**: Root Mean Squared Error (lower is better)

Example output:
```
LINEAR     | R²: 0.1234 | RMSE: 0.05123
RF         | R²: 0.2456 | RMSE: 0.04891
XGBOOST    | R²: 0.3187 | RMSE: 0.04567
```

## On Model Performance and Market Efficiency

XGBoost achieves R²=0.31 on held-out test data. We interpret this not as a failure but as empirical evidence consistent with the semi-strong form of the Efficient Market Hypothesis — technical indicators computed from public price data contain limited incremental predictive signal beyond what is already priced in.

The value of this framework is not alpha generation but infrastructure: a systematic pipeline for testing whether any feature set violates EMH, with proper train/test isolation and walk-forward backtesting to prevent look-ahead bias. That reframe turns a mediocre ML result into a principled research finding. It's also true.

## 📔 Jupyter Notebooks

Explore analysis and workflows:

- `data_fetching.ipynb` - Downloading and cleaning stock data
- `feature_engineering.ipynb` - Technical indicator construction
- `model_training.ipynb` - Model training and hyperparameter tuning
- `backtest_and_analysis.ipynb` - Strategy backtesting and performance analysis

## ⚙️ Configuration

Modify settings in `src/datapipeline.py`:

```python
USE = "SP500"        # Switch to "NIFTY50" for Indian stocks
```

## 🔍 Explainability

Interpret model predictions:

```python
from src.explainability import explain_prediction, feature_importance

# Feature importance
importance = feature_importance(models["xgboost"], X_test)
print(importance)

# Individual prediction explanation
explanation = explain_prediction(
    model=models["xgboost"],
    sample=X_test.iloc[0],
    method="shap"
)
```

## ⚠️ Disclaimer

**This project is for educational and research purposes only.** Financial markets are complex and unpredictable. Past performance does not guarantee future results. Never use algorithmic predictions as the sole basis for investment decisions. Always conduct thorough due diligence and consult with financial advisors before trading.

## 📚 Dependencies

Core libraries:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML models and metrics
- `xgboost` - Gradient boosting
- `yfinance` - Stock data fetching
- `ta-lib` - Technical analysis indicators
- `streamlit` - Web dashboard

See `requirements.txt` for complete list.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models (LSTM, Transformer)
- More technical indicators
- Risk management modules
- Real-time data pipelines
- API integration (Alpha Vantage, IEX Cloud)

## 📝 License

[Add your license here]

## 🔗 References

- [Technical Analysis Library (TA-Lib)](https://technical-analysis-library-in-python.readthedocs.io/)
- [YFinance Documentation](https://yfinance.readthedocs.io/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## ✉️ Contact

For questions or feedback, please open an issue or contact the development team.

---

**Last Updated**: March 2026  
**Status**: Active Development
