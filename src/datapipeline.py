import pandas as pd

USE = "SP500"   # switch to "NIFTY50" later if needed

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    return tables[0]['Symbol'].tolist()

def get_nifty50_tickers():
    return ["RELIANCE.NS","HDFCBANK.NS","TCS.NS","INFY.NS","ICICIBANK.NS"]

TICKERS = get_sp500_tickers() if USE == "SP500" else get_nifty50_tickers()
print(f"Universe: {USE}  |  Count: {len(TICKERS)}")
