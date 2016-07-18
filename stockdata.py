import numpy as np
import pandas as pd
import os
import urllib.request


# List of indices we are interested in
INDEX_LIST = ["N225",   # Nikkei 225, Japan
               "HSI",   # Hang Seng, Hong Kong
               "AORD",  # All Ords, Australia
               "STI",   # STI Index, Singapore
               "GDAXI", # DAX, German
               "FTSE",  # FTSE 100, UK
               "DJI",   # Dow, US
               "GSPC",  # S&P 500, US
               "IXIC",  # NASDAQ, US
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

        self.baseurl = r"http://real-chart.finance.yahoo.com/table.csv?ignore=.csv&s=%5E"
        self.baseurl2 = r"http://www.m2j.co.jp/market/pchistry_dl.php?type=d&ccy="

    def download(self):
        """Download historical data for major markets to "./data/<code>.csv"
        """
        # Stock markets
        for index in INDEX_LIST:
            filename = self.basedir + index + ".csv"
            url = self.baseurl + index
            try:
                with urllib.request.urlopen(urllib.request.Request(url)) as response:
                    message = response.read()
                    with open(filename, "wb") as f:
                        f.write(message)
            except:
                print("Error:", url)

        # Exchange rates
        for pair in CURRENCY_PAIR_LIST:
            filename = self.basedir + pair + ".csv"
            url = self.baseurl2 + str(CURRENCY_PAIR_TO_ID[pair])
            try:
                with urllib.request.urlopen(urllib.request.Request(url)) as response:
                    message = response.read()
                    with open(filename, "wb") as f:
                        f.write(message)
            except Exception as e:
                print("Error:", url)
                print(e)


    def get_closing_data(self, days=1000, normalize=True, logreturn=True):
        """ Get closing data for indecis.

        If days is given, return the data for last <days> days.
        If normalize is set to True, then closing value is normalized during the period.
        :type days: int
        :type normalize: bool
        :type logscale: bool
        :rtype: pandas.DataFrame
        """
        today = np.datetime64("today")
        start_date = today - np.timedelta64(days, 'D')
        stock_data = self.get_stock_market_closing_data(start_date, today, normalize, logreturn)
        exchange_rate = self.get_exchange_rate_closing_data(start_date, today, normalize, logreturn)

        closing_data = pd.DataFrame()
        closing_data['Date'] = pd.date_range(start_date, periods=1000, freq='D')
        closing_data = closing_data.set_index("Date")
        for index in stock_data.columns:
            closing_data[index] = stock_data[index]
        for pair in exchange_rate.columns:
            closing_data[pair] = exchange_rate[pair]

        # TODO: fillnaはここでやるべき
        closing_data = closing_data[::-1].fillna(method="ffill")[::-1].fillna(method="ffill")
        return closing_data

    def get_stock_market_closing_data(self, start_date, end_date, normalize=True, logreturn=True):
        """ Get closing data for indecis.

        If days is given, return the data for last <days> days.
        If normalize is set to True, then closing value is normalized during the period.
        :type days: int
        :type normalize: bool
        :type logscale: bool
        :rtype: pandas.DataFrame
        """
        closing_data = pd.DataFrame()
        for index in INDEX_LIST:
            df = pd.read_csv(self.basedir + index + ".csv")
            df["Date"] = pd.to_datetime(df["Date"])
            mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
            df = df.loc[mask].set_index("Date")
            closing_data[index] = df["Close"]

        # Reverse the dataframe as CSV contains data in desc order
        # Also, fill empty cells by fillna method
        closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill")

        # Normalizations
        for index in INDEX_LIST:
            if normalize:
                closing_data[index] = closing_data[index] / max(closing_data[index])
            if logreturn:
                closing_data[index] = np.log(closing_data[index] / closing_data[index].shift())
        return closing_data

    def get_exchange_rate_closing_data(self, start_date, end_date, normalize=True, logreturn=True):
        closing_data = pd.DataFrame()
        for pair in CURRENCY_PAIR_LIST:
            filename = self.basedir + pair + ".csv"
            df = pd.read_csv(filename, header=0, names=("Date", "Start", "High", "Low", "Close"), encoding="sjis")
            df["Date"] = pd.to_datetime(df["Date"])
            mask = (df['Date'] > start_date) & (df['Date'] <= end_date)
            df = df.loc[mask].set_index("Date")
            df = df[::-1]
            closing_data[pair] = df["Close"]

        # Reverse the dataframe as CSV contains data in desc order
        # Also, fill empty cells by fillna method
        closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill")

        # Normalizations
        for pair in CURRENCY_PAIR_LIST:
            if normalize:
                closing_data[pair] = closing_data[pair] / max(closing_data[pair])
            if logreturn:
                closing_data[pair] = np.log(closing_data[pair] / closing_data[pair].shift())
        return closing_data


class ExchangeRateData(object):
    def __init__(self):
        self.basedir = "./data/"
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        self.baseurl = r"http://www.m2j.co.jp/market/pchistry_dl.php?type=d&ccy="

    def download(self):
        """Download historical data for exchange rates to "./data/<pair>.csv"
        """
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

    def get_closing_data(self, days=1000, normalize=True, logreturn=True):
        closing_data = pd.DataFrame()
        for pair in CURRENCY_PAIR_LIST:
            filename = self.basedir + pair + ".csv"
            df = pd.read_csv(filename, header=0, names=("Date", "Start", "High", "Low", "Close"), encoding="sjis").set_index("Date")
            df = df[::-1]
            closing_data[pair] = df["Close"][:days] if days > 0 else df["Close"]

        # Reverse the dataframe as CSV contains data in desc order
        # Also, fill empty cells by fillna method
        closing_data = closing_data.fillna(method="ffill")[::-1].fillna(method="ffill")

        # Normalizations
        for pair in CURRENCY_PAIR_LIST:
            if normalize:
                closing_data[pair] = closing_data[pair] / max(closing_data[pair])
            if logreturn:
                closing_data[pair] = np.log(closing_data[pair] / closing_data[pair].shift())
        return closing_data


# e = ExchangeRateData()
# e.get_closing_data(normalize=True, logreturn=True)
