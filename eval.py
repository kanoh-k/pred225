import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from stockdata import StockData, INDEX_LIST, INDEX_TO_DISPLAY_TEXT, CURRENCY_PAIR_LIST
from model import *

def how_to_use():
    # Set date range for training/test data
    train_start_date = np.datetime64("2012-01-01")
    train_end_date = np.datetime64("2015-12-31")
    test_start_date = np.datetime64("2016-01-01")
    test_end_date = np.datetime64("2016-08-31")

    # Doanload latest data
    stockdata = StockData()
    stockdata.download()

    # How to train
    model = NikkeiModel([], "YourModelName")
    model.prepare_training_data(train_start_date, train_end_date)
    model.train()

    # How to evaluate
    model.prepare_test_data(test_start_date, test_end_date)
    model.evaluate()
    model.backtest()

    # How to predict
    n225_open = 16500 # Today's N225 Open value
    model.predict(n225_open, np.datetime64("today"), downloadData=False)

def run_backtests():
    test_start_date = np.datetime64("2016-01-01")
    test_end_date = np.datetime64("2016-08-28")

    train_period = [
        ("2008-01-01", "2015-12-31"),
        ("2009-01-01", "2015-12-31"),
        ("2010-01-01", "2015-12-31"),
        ("2011-01-01", "2015-12-31"),
        ("2012-01-01", "2015-12-31"),
        ("2013-01-01", "2015-12-31"),
        ("2014-01-01", "2015-12-31"),
        ("2015-01-01", "2015-12-31"),
        ("2013-01-01", "2015-08-26")
    ]

    # Training
    for start, end in train_period:
        filename = "%s_%s_25-50-25-3" % (start, end)
        model = NikkeiModel([50, 25], "./model/" + filename)
        if not model.is_trained():
            train_start_date = np.datetime64(start)
            train_end_date = np.datetime64(end)
            model.prepare_training_data(train_start_date, train_end_date)
            model.train()

    # Evaluation
    for start, end in train_period:
        filename = "%s_%s_25-50-25-3" % (start, end)
        model = NikkeiModel([50, 25], "./model/" + filename)
        model.backtest_result = "./model/test_" + filename + ".csv"
        print("******************************\n")
        print("Evaluation: %s - %s\n" % (start, end))
        print("******************************\n")
        model.prepare_test_data(test_start_date, test_end_date)
        model.evaluate()
        model.backtest()

def compare_different_layers():
    train_start_date = np.datetime64("2010-01-01")
    train_end_date = np.datetime64("2014-12-31")
    test_start_date = np.datetime64("2015-01-01")
    test_end_date = np.datetime64("2016-07-31")

    layers_list = [
        [],
        [25],
        [50],
        [100],
        [25, 10],
        [40, 20],
        [50, 25],
        [75, 30],
        [100, 50],
        [50, 25, 10],
        [100, 50, 25],
        [200, 100, 50]
    ]

    for layers in layers_list:
        model = NikkeiModel(layers, './model/2010-2015_37-' + '-'.join([str(x) for x in layers]) + '-3')
        if not model.is_trained():
            model.prepare_training_data(train_start_date, train_end_date)
            model.prepare_test_data(test_start_date, test_end_date)
            model.train(eval_on_test=True)

def run_backtest():
    test_start_date = np.datetime64("2015-01-01")
    test_end_date = np.datetime64("2016-07-31")
    model = NikkeiModel([50, 25], './model/2010-2015_37-50-25-3')
    model.prepare_test_data(test_start_date, test_end_date)
    model.evaluate()
    model.backtest()

if __name__ == "__main__":
    compare_different_layers()
    # run_backtest()
