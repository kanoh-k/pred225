import numpy as np
import pandas as pd
import os
import urllib.request
import quandl

quandl.ApiConfig.api_key = "Your API Key"
quandl.ApiConfig.api_version = "2015-04-09"

# List of indices we are interested in
INDEX_LIST = ["N225",   # Nikkei 225, Japan
               "HSI",   # Hang Seng, Hong Kong
               "AORD",  # All Ords, Australia
               "STI",   # STI Index, Singapore
               "GDAXI", # DAX, German
#               "FTSE",  # FTSE 100, UK
               "DJI",   # Dow, US
               "GSPC",  # S&P 500, US
#               "IXIC",  # NASDAQ, US
               "BVSP"]  # BOVESPA, Brazil

# Mapping from an index to its display text
INDEX_TO_DISPLAY_TEXT = {
    "N225": "Nikkei",
    "HSI": "Hang Seng",
    "AORD": "All Ords",
    "STI": "STI",
    "GDAXI": "DAX",
    "FTSE": "FTSE 100",
    "DJI": "Dow",
    "GSPC": "S&P 500",
    "IXIC": "NASDAQ",
    "BVSP": "BOVESPA"
}

CURRENCY_PAIR_LIST = ["USD-JPY", "EUR-JPY"]
CURRENCY_PAIR_TO_ID = {
    "USD-JPY": 1,
    "EUR-JPY": 2
}

class StockData(object):
    def __init__(self):
        self.basedir = "./data/"
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        self.baseurl = r"http://www.m2j.co.jp/market/pchistry_dl.php?type=d&ccy="

    def download(self):
        """Download historical data for major markets to "./data/<code>.csv"
        """
        # Stock markets
        for index in INDEX_LIST:
            filename = self.basedir + index + ".csv"
            try:
                data = quandl.get("YAHOO/INDEX_" + index)
                data.to_csv(filename)
            except:
                print("Error: failed to download" + index)

        # Exchange rates
        for pair in CURRENCY_PAIR_LIST:
            filename = self.basedir + pair + ".csv"
            url = self.baseurl + str(CURRENCY_PAIR_TO_ID[pair])
            try:
                with urllib.request.urlopen(urllib.request.Request(url)) as response:
                    message = response.read()
                    with open(filename, "wb") as f:
                        f.write(message)
            except Exception as e:
                print("Error:", url)
                print(e)

    def get_daily_data(self, days=1000, normalize=True, logreturn=True, start_date=None, end_date=None, fill_empty=True):
        """ Get closing data for indecis.

        If days is given, return the data for last <days> days.
        If normalize is set to True, then closing value is normalized during the period.
        :type days: int
        :type normalize: bool
        :type logscale: bool
        :rtype: pandas.DataFrame
        """
        end_date = np.datetime64("today") if end_date is None else end_date
        start_date = end_date - np.timedelta64(days, 'D') if start_date is None else start_date
        stock_data = self.get_stock_market_daily_data(start_date, end_date, normalize, logreturn, fill_empty=fill_empty)
        exchange_rate = self.get_exchange_rate_daily_data(start_date, end_date, normalize, logreturn, fill_empty=fill_empty)

        opening_data, closing_data, diff_data, jump_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        opening_data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
        closing_data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
        diff_data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
        jump_data['Date'] = pd.date_range(start=start_date, end=end_date, freq='D')
        opening_data = opening_data.set_index("Date")
        closing_data = closing_data.set_index("Date")
        diff_data = diff_data.set_index("Date")
        jump_data = jump_data.set_index("Date")

        for index in stock_data[0].columns:
            opening_data[index] = stock_data[0][index]
        for index in stock_data[1].columns:
            closing_data[index] = stock_data[1][index]
        for index in stock_data[2].columns:
            diff_data[index] = stock_data[2][index]
        for index in stock_data[3].columns:
            jump_data[index] = stock_data[3][index]
        for pair in exchange_rate[0].columns:
            opening_data[pair] = exchange_rate[0][pair]
        for pair in exchange_rate[1].columns:
            closing_data[pair] = exchange_rate[1][pair]
        for pair in exchange_rate[2].columns:
            diff_data[pair] = exchange_rate[2][pair]
        for pair in exchange_rate[3].columns:
            jump_data[pair] = exchange_rate[3][pair]

        # TODO: fillnaはここでやるべき
        if fill_empty:
            opening_data = opening_data[::-1].fillna(method="ffill")[::-1].fillna(method="ffill")
            closing_data = closing_data[::-1].fillna(method="ffill")[::-1].fillna(method="ffill")
            diff_data = diff_data[::-1].fillna(method="ffill")[::-1].fillna(method="ffill")
            jump_data = jump_data[::-1].fillna(method="ffill")[::-1].fillna(method="ffill")
        return opening_data, closing_data, diff_data, jump_data

    def get_stock_market_daily_data(self, start_date, end_date, normalize=True, logreturn=True, fill_empty=True):
        """ Get closing data for indecis.

        If days is given, return the data for last <days> days.
        If normalize is set to True, then closing value is normalized during the period.
        :type days: int
        :type normalize: bool
        :type logscale: bool
        :rtype: pandas.DataFrame
        """
        opening_data, closing_data = pd.DataFrame(), pd.DataFrame()
        for index in INDEX_LIST:
            df = pd.read_csv(self.basedir + index + ".csv")
            df["Date"] = pd.to_datetime(df["Date"])
            mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
            df = df.loc[mask].set_index("Date")
            opening_data[index] = df["Open"]
            closing_data[index] = df["Close"]

        # Reverse the dataframe as CSV contains data in desc order
        # Also, fill empty cells by fillna method
        if fill_empty:
            opening_data = opening_data.fillna(method="ffill")[::-1].fillna(method="ffill")
            closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill")
        diff_data = closing_data / opening_data - 1
        jump_data = opening_data / closing_data.shift() - 1

        # Normalizations
        for index in INDEX_LIST:
            if normalize:
                opening_data[index] = opening_data[index] / max(opening_data[index])
                closing_data[index] = closing_data[index] / max(closing_data[index])
            if logreturn:
                opening_data[index] = np.log(opening_data[index] / opening_data[index].shift())
                closing_data[index] = np.log(closing_data[index] / closing_data[index].shift())
        return opening_data, closing_data, diff_data, jump_data

    def get_exchange_rate_daily_data(self, start_date, end_date, normalize=True, logreturn=True, fill_empty=True):
        opening_data, closing_data = pd.DataFrame(), pd.DataFrame()
        for pair in CURRENCY_PAIR_LIST:
            filename = self.basedir + pair + ".csv"
            df = pd.read_csv(filename, header=0, names=("Date", "Open", "High", "Low", "Close"), encoding="sjis")
            df["Date"] = pd.to_datetime(df["Date"])
            mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
            df = df.loc[mask].set_index("Date")
            df = df[::-1]
            opening_data[pair] = df["Open"]
            closing_data[pair] = df["Close"]

        # Reverse the dataframe as CSV contains data in desc order
        # Also, fill empty cells by fillna method
        if fill_empty:
            opening_data = opening_data.fillna(method="ffill")[::-1].fillna(method="ffill")
            closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill")
        diff_data = closing_data / opening_data - 1
        jump_data = opening_data / closing_data.shift() - 1

        # Normalizations
        for pair in CURRENCY_PAIR_LIST:
            if normalize:
                opening_data[pair] = opening_data[pair] / max(opening_data[pair])
                closing_data[pair] = closing_data[pair] / max(closing_data[pair])
            if logreturn:
                opening_data[pair] = np.log(opening_data[pair] / opening_data[pair].shift())
                closing_data[pair] = np.log(closing_data[pair] / closing_data[pair].shift())
        return opening_data, closing_data, diff_data, jump_data
