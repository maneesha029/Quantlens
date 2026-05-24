import pandas as pd
import numpy as np
import ta

def generate_features(df):
    df = df.copy()

    # yfinance may return a DataFrame with MultiIndex columns when a single
    # ticker is downloaded (e.g. ('Close','AAPL')). Normalize to single
    # level columns when the second level contains a single ticker.
    if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
        try:
            second_level = df.columns.get_level_values(1)
            if second_level.nunique() == 1:
                df.columns = df.columns.get_level_values(0)
            else:
                # If multiple tickers present, user should pass a single ticker
                raise ValueError("DataFrame contains multiple tickers; pass a single-symbol DataFrame to generate_features().")
        except Exception:
            # Fallback: attempt to squeeze columns like Close/High/Low individually below
            pass

    # Technical indicators
    # Ensure inputs to ta functions are 1D Series (squeeze if needed)
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["macd"] = ta.trend.MACD(close).macd()
    df["macd_signal"] = ta.trend.MACD(close).macd_signal()
    df["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close).stoch()
    df["stoch_d"] = ta.momentum.StochasticOscillator(high, low, close).stoch_signal()

    # Rolling features
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["volatility_20"] = df["Close"].pct_change().rolling(20).std()
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    # Target variable: next 5-day return
    df["target_5d"] = df["Close"].shift(-5) / df["Close"] - 1

    # Remove lookahead bias
    df.dropna(inplace=True)

    return df
