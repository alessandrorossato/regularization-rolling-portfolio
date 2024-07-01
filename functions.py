import pandas as pd
import numpy as np
import seaborn as sns

def get_windows(stocks_roll, sp500_roll, n):
    # take the price of the stocks in the window
    portfolio_index = sp500_roll[n]['Adj Close'].dropna().to_frame()
    portfolio_stocks = stocks_roll[n].ffill(limit=2).dropna(axis='columns')
    
    # scale the data
    portfolio_index = portfolio_index.div(portfolio_index.iloc[0])
    portfolio_stocks = portfolio_stocks.div(portfolio_stocks.iloc[0])
    
    return portfolio_index, portfolio_stocks


def split_windows(portfolio_index, portfolio_stocks, perc=0.8):
    # years_window = train+test
    # split the data into training and testing
    train_index = portfolio_index.iloc[:int(len(portfolio_index)*perc)]
    test_index = portfolio_index.iloc[int(len(portfolio_index)*perc):]
    
    train_stocks = portfolio_stocks.iloc[:int(len(portfolio_stocks)*perc)]
    test_stocks = portfolio_stocks.iloc[int(len(portfolio_stocks)*perc):]
    
    return train_index, test_index, train_stocks, test_stocks


def risk_free(train_stocks, test_stocks, download=False, years=5):
    # temporal index
    temp_frame = train_stocks.index
    train_date = (temp_frame[0]+pd.DateOffset(day=1)).date()
    temp_frame = test_stocks.index
    test_date = (temp_frame[0]+pd.DateOffset(day=1)).date()

    if download:
        from fredapi import Fred
        fred = Fred(api_key='20e590f146e49b1b938cc37d6c3a3266')
        hist_risk_free = fred.get_series_latest_release(f'GS{years}')/100 # GS5 or GS10
        hist_risk_free.index = pd.to_datetime(hist_risk_free.index).date
        hist_risk_free.to_csv(f'Data/{years}_year_yield.csv')
    else:
        hist_risk_free = pd.read_csv(f'Data/{years}_year_yield.csv', index_col=0, parse_dates=True)
        hist_risk_free.index = hist_risk_free.index.date
        
    risk_free_train = hist_risk_free.loc[train_date:train_date,]
    risk_free_test = hist_risk_free.loc[test_date:test_date,]
    
    return risk_free_train, risk_free_test

def returns(df_price):
        df_ret = df_price.ffill(limit=2).dropna()
        
        # returns
        df_ret = df_ret.pct_change().dropna()

        # find the stock with returns higher than 50%
        max_return = df_ret.max().sort_values(ascending=False)
        max_return = max_return[max_return > 0.50]
        max_return = max_return.index   
        df_ret = df_ret.drop(max_return, axis=1)

        return df_ret

def sectors(equities_sector, returns_stocks):
    # create a new dict with ticker: sector
    sector_dict = {}
    for sector, tickers in equities_sector.items():
        for ticker in tickers:
            sector_dict[ticker] = sector
            
    sector_list = []
    # extract sector from dict equities_sector 
    for colname in returns_stocks.columns:
        sector = sector_dict.get(colname)
        sector_list.append(sector)
        
    sector_list = list(set(sector_list))

    # color palette for the sectors
    palette = sns.color_palette("tab10", len(sector_list))
    
    return sector_dict, sector_list, palette

def sharpe_ratio(final_ret, std, risk_free):
    sharpe = (final_ret-risk_free)/std
    return sharpe

def max_drawdown(cum_ret):
    max_drawdown = (cum_ret.cummax()-cum_ret).max()
    return max_drawdown