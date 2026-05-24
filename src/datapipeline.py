import pandas as pd

USE = "SP500"   # switch to "NIFTY50" later if needed

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        return tables[0]['Symbol'].tolist()
    except Exception as e:
        # If pandas.read_html or lxml is unavailable or the network fails,
        # return an empty list and warn the user. This prevents import-time
        # failures when the environment lacks optional deps.
        print(f"Warning: could not fetch S&P 500 tickers: {e}")
        return []

def get_nifty50_tickers():
    return ["RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS","ICICIBANK.NS"]

TICKERS = get_sp500_tickers() if USE == "SP500" else get_nifty50_tickers()
print(f"Universe: {USE}  |  Count: {len(TICKERS)}")
