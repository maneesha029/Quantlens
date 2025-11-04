import pandas as pd
import numpy as np

def backtest_model(df, model, top_n=5, initial_capital=100000):
    """
    Simple backtest using predicted next 5-day returns
    df: processed features (multiindex: Ticker, Date)
    model: trained ML model
    top_n: number of stocks to invest in each day
    initial_capital: starting cash
    """
    df = df.copy()
    df["pred_return"] = model.predict(df.drop(columns=["target_5d"]))
    
    # Pivot to Date x Ticker
    df_reset = df.reset_index()
    dates = sorted(df_reset["Date"].unique())
    tickers = df_reset["Ticker"].unique()
    
    portfolio_value = []
    benchmark_value = []
    capital = initial_capital
    
    for date in dates:
        day_df = df_reset[df_reset["Date"] == date].sort_values("pred_return", ascending=False)
        top_stocks = day_df["Ticker"].values[:top_n]
        weights = np.repeat(1/top_n, top_n)
        
        # Compute returns
        day_returns = day_df[day_df["Ticker"].isin(top_stocks)]["target_5d"].values
        day_return = np.dot(weights, day_returns)
        
        capital = capital * (1 + day_return)
        portfolio_value.append(capital)
        
        # Benchmark: equal weight across all tickers
        benchmark_return = day_df["target_5d"].mean()
        if len(benchmark_value) == 0:
            benchmark_value.append(initial_capital * (1 + benchmark_return))
        else:
            benchmark_value.append(benchmark_value[-1] * (1 + benchmark_return))
    
    portfolio = pd.DataFrame({
        "Date": dates,
        "Portfolio_Value": portfolio_value,
        "Benchmark_Value": benchmark_value
    })
    
    portfolio.set_index("Date", inplace=True)
    return portfolio
