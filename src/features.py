import pandas as pd
import numpy as np
import ta

def generate_features(df):
    df = df.copy()

    # Technical indicators
    df["rsi_14"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal()
    df["stoch_k"] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"]).stoch()
    df["stoch_d"] = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"]).stoch_signal()

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
