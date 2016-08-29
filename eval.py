import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from stockdata import StockData, INDEX_LIST, INDEX_TO_DISPLAY_TEXT, CURRENCY_PAIR_LIST
from model import *

def evaluate_model(intermediate_layers=[]):
    training_predictors_tf, training_classes_tf, test_predictors_tf, test_classes_tf = create_training_test_data()

    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)

    feature_data = tf.placeholder("float", [None, num_predictors])
    labels = tf.placeholder("float", [None, num_classes])

    layers = [num_predictors] + intermediate_layers + [num_classes]
    experiment_name = '-'.join([str(n) for n in layers])

    train_dict = {
        feature_data: training_predictors_tf.values,
        labels: training_classes_tf.values.reshape(len(training_classes_tf.values), num_classes)}

    test_dict = {
        feature_data: test_predictors_tf.values,
        labels: test_classes_tf.values.reshape(len(test_classes_tf.values), num_classes)}

    with tf.Session() as sess:
        model = inference(feature_data, layers)
        cost, cost_summary_op = loss(model, labels)
        training_op = training(cost, learning_rate=0.0001)

        correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accuracy_op_train = tf.scalar_summary("Accuracy on Train", accuracy)
        accuracy_op_test = tf.scalar_summary("Accuracy on Test", accuracy)

        # Merge all variable summaries and save the results to log file
        # summary_op = tf.merge_all_summaries()
        summary_op_train = tf.merge_summary([cost_summary_op, accuracy_op_train])
        summary_op_test = tf.merge_summary([accuracy_op_test])
        summary_writer = tf.train.SummaryWriter("/tmp/pred225_log3/" + experiment_name, sess.graph)

        init = tf.initialize_all_variables()
        sess.run(init)

        for i in range(1, 30001):
            sess.run(training_op, feed_dict=train_dict)

            # Write summary to log
            if i % 100 == 0:
                # summary_str = sess.run(summary_op, feed_dict=train_dict)
                summary_str = sess.run(summary_op_train, feed_dict=train_dict)
                summary_writer.add_summary(summary_str, i)
                summary_str = sess.run(summary_op_test, feed_dict=test_dict)
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()

            # Print current accuracy to console
            if i%5000 == 0:
                print (i, sess.run(accuracy, feed_dict=train_dict))

        # tf_confusion_metrics(model, labels, sess, test_dict)
        evaluate(model, labels, sess, test_dict)

        # Save model
        saver = tf.train.Saver()
        saver.save(sess, "model.ckpt")

def evaluate_models():
    experiments = [
        # []#,
        # [25],
        # [50],
        # [100],
        [50, 25]
        # [100, 50],
        # [50, 25, 10],
        # [100, 50, 25],
        # [200, 100, 50]
    ]

    today = np.datetime64("today")
    train_start_date = today - np.timedelta64(1007, 'D')
    train_end_date = today - np.timedelta64(208, 'D')
    test_start_date = today - np.timedelta64(207, 'D')
    test_end_date = today - np.timedelta64(7, 'D')

    for layers in experiments:
        # evaluate_model(layers)
        train_and_test(train_start_date, train_end_date, test_start_date, test_end_date, layers)

def eval0828():

    model = NikkeiModel([50, 25], "2013-201607_28-50-25-2")
    # model.prepare_training_data(train_start_date, train_end_date)
    # model.train()
    # model.prepare_test_data(test_start_date, test_end_date)
    model.prepare_test_data(test_start_date, test_end_date)
    model.evaluate()
    model.backtest()

    # model.predict(np.datetime64("2016-08-18"), downloadData=False)

    # model.prepare_test_data(test_start_date, test_end_date)
    # model.backtest_monkey()

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

    eval0828()

if __name__ == "__main__":
    how_to_use()
