import numpy as np
import pandas as pd
import tensorflow as tf
import random
from datetime import datetime
from tensorflow.python.framework import ops
from stockdata import StockData, INDEX_LIST, INDEX_TO_DISPLAY_TEXT, CURRENCY_PAIR_LIST

class NikkeiModel:
    def __init__(self, hidden_layers, model_name):
        self.max_iteration = 30001
        self.threshold = 0.003
        self.log_dir = "/tmp/pred225_log3/"
        self.backtest_result = "test.csv"

        self.stockdata = StockData()
        self.hidden_layers = hidden_layers
        self.model_filename = model_name + ".ckpt"

        self.training_predictors_tf = None
        self.training_classes_tf = None
        self.test_predictors_tf = None
        self.test_classes_tf = None
        self.raw_values = None
        self.num_predictors = None
        self.num_classes = None

    def inference(self, feature_data, layers):
        """ Create a model with given numbers of layers.

        'layers' is the list of integers. Each element in the list defines the number
        of nodes in each layer. layers[0] is an input layer, layers[-1] is an output
        layer. Other layers[i] means i-th hidden layer.
        :type feature_data: ???
        :type layers: List<int>
        """
        if len(layers) < 2:
            raise Exception("Layer is invalid")

        previous_layer = feature_data
        for i in range(len(layers) - 2):
            with tf.name_scope("Hidden" + str(i + 1)):
                weights = tf.Variable(tf.truncated_normal([layers[i], layers[i + 1]], stddev=0.0001))
                biases = tf.Variable(tf.ones([layers[i + 1]]))
                previous_layer = tf.nn.relu(tf.matmul(previous_layer, weights) + biases)

        with tf.name_scope("Output"):
            weights = tf.Variable(tf.truncated_normal([layers[-2], layers[-1]], stddev=0.0001))
            biases = tf.Variable(tf.ones([layers[-1]]))
            model = tf.nn.softmax(tf.matmul(previous_layer, weights) + biases)
            return model

    def loss(self, model, labels):
        """ Loss function and its scalar summary string

        :type model: ??
        :type labels: ??
        :rtype: (??, string)
        """
        cost = -tf.reduce_sum(labels*tf.log(model))
        return cost ,tf.scalar_summary("Cross entropy", cost)

    def training(self, cost, learning_rate=0.0001):
        """ Training part
        """
        training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        return training_op

    def create_data(self, start_date, end_date):
        opening_data, closing_data, diff_data, jump_data = self.stockdata.get_daily_data(start_date=start_date, end_date=end_date)
        opening_raw, closing_raw, diff_raw, _ = self.stockdata.get_daily_data(start_date=start_date, end_date=end_date, normalize=False, logreturn=False, fill_empty=False)

        closing_data["N225_up"] = 0
        closing_data.ix[diff_data["N225"] >= self.threshold, "N225_up"] = 1
        closing_data["N225_down"] = 0
        closing_data.ix[diff_data["N225"] <= -self.threshold, "N225_down"] = 1
        closing_data["N225_same"] = 0
        closing_data.ix[(-self.threshold < diff_data["N225"]) & (diff_data["N225"] < self.threshold), "N225_same"] = 1

        columns = ["N225_up", "N225_down", "N225_same"] + ["N225_jump"]\
        + [s + "_1_o" for s in INDEX_LIST[1:]] + [s + "_1_o" for s in CURRENCY_PAIR_LIST] \
        + [s + "_1_c" for s in INDEX_LIST[1:]] + [s + "_1_c" for s in CURRENCY_PAIR_LIST] \
        + [s + "_1_d" for s in INDEX_LIST[1:]] + [s + "_1_d" for s in CURRENCY_PAIR_LIST]

        training_test_data = pd.DataFrame(
            # column name is "<index>_<day>_<type>".
            # E.g., "DJI_1_c" means yesterday's Dow closing value.
            columns=columns
        )

        for i in range(len(closing_data)):
            data = {}
            # We will use today's data for positive/negative labels
            data["N225_up"] = closing_data["N225_up"].ix[i]
            data["N225_down"] = closing_data["N225_down"].ix[i]
            data["N225_same"] = closing_data["N225_same"].ix[i]
            data["N225_jump"] = jump_data["N225"].ix[i]

            # Use yesterday's data for world market data
            for col in INDEX_LIST[1:] + CURRENCY_PAIR_LIST:
                data[col + "_1_o"] = closing_data[col].ix[i - 1]
                data[col + "_1_c"] = opening_data[col].ix[i - 1]
                data[col + "_1_d"] = diff_data[col].ix[i - 1]

            # Date and its actual Open/Close values for back-test
            data["N225_open"] = opening_raw["N225"].ix[i]
            data["N225_close"] = closing_raw["N225"].ix[i]
            data["N225_diff"] = diff_raw["N225"].ix[i]
            data["Date"] = diff_raw["N225"].index[i]

            training_test_data = training_test_data.append(data, ignore_index=True)

        # Prepare training data and test data
        predictors_tf = training_test_data[training_test_data.columns[3:-4]]
        classes_tf = training_test_data[training_test_data.columns[:3]]
        raw_values = training_test_data[training_test_data.columns[-4:]]

        if self.num_predictors is None:
            self.num_predictors = len(predictors_tf.columns)
            self.num_classes = len(classes_tf.columns)

        return predictors_tf, classes_tf, raw_values

    def prepare_training_data(self, start_date, end_date):
        """ Load training data. Must be called before train().
        """
        self.training_predictors_tf, self.training_classes_tf, _ = self.create_data(start_date, end_date)

    def prepare_test_data(self, start_date, end_date):
        """ Load test data. Must be called before evaluate().
        """
        self.test_predictors_tf, self.test_classes_tf, self.raw_values = self.create_data(start_date, end_date)

    def train(self):
        """ Train model and save it to file.

        Train model with given hidden layers. Training data is created
        by prepare_training_data(), which must be called before this function.
        """
        with tf.Session() as sess:
            feature_data = tf.placeholder("float", [None, self.num_predictors])
            labels = tf.placeholder("float", [None, self.num_classes])

            layers = [self.num_predictors] + self.hidden_layers + [self.num_classes]
            experiment_name = '-'.join([str(n) for n in layers])
            model = self.inference(feature_data, layers)
            cost, cost_summary_op = self.loss(model, labels)
            training_op = self.training(cost, learning_rate=0.0001)

            correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_op_train = tf.scalar_summary("Accuracy on Train", accuracy)

            # Merge all variable summaries and save the results to log file
            summary_op = tf.merge_all_summaries()
            # summary_op_train = tf.merge_summary([cost_summary_op, accuracy_op_train])
            # summary_op_test = tf.merge_summary([accuracy_op_test])
            summary_writer = tf.train.SummaryWriter(self.log_dir + experiment_name, sess.graph)

            train_dict = {
                feature_data: self.training_predictors_tf.values,
                labels: self.training_classes_tf.values.reshape(len(self.training_classes_tf.values), self.num_classes)}

            init = tf.initialize_all_variables()
            sess.run(init)

            for i in range(1, self.max_iteration):
                sess.run(training_op, feed_dict=train_dict)

                # Write summary to log
                if i % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict=train_dict)
                    summary_writer.add_summary(summary_str, i)
                    summary_writer.flush()

                # Print current accuracy to console
                if i%5000 == 0:
                    print (i, sess.run(accuracy, feed_dict=train_dict))

            # Save trained parameters
            saver = tf.train.Saver()
            saver.save(sess, self.model_filename)

    def evaluate(self):
        """ Restore trained model and evaluate it.

        Restore model from self.model_filename, and evaluate its accuracy.
        Test data is created by prepare_test_data(), which must be called
        before this function.
        """
        tf.reset_default_graph()
        with tf.Session() as sess:
            feature_data = tf.placeholder("float", [None, self.num_predictors])
            labels = tf.placeholder("float", [None, self.num_classes])
            layers = [self.num_predictors] + self.hidden_layers + [self.num_classes]
            model = self.inference(feature_data, layers)
            feed_dict = {
                feature_data: self.test_predictors_tf.values,
                labels: self.test_classes_tf.values.reshape(len(self.test_classes_tf.values), self.num_classes)}

            # Restore parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.model_filename)

            predictions = tf.argmax(model, 1)
            actuals = tf.argmax(labels, 1)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_actuals = tf.ones_like(actuals)
            twos_like_actuals = tf.scalar_mul(2, tf.ones_like(actuals))
            zeros_like_predictions = tf.zeros_like(predictions)
            ones_like_predictions = tf.ones_like(predictions)
            twos_like_predictions = tf.scalar_mul(2, tf.ones_like(predictions))

            # Count # of samples in each class
            up_actuals_count_op = tf.reduce_sum(tf.cast(tf.equal(actuals, zeros_like_actuals), "float"))
            down_actuals_count_op = tf.reduce_sum(tf.cast(tf.equal(actuals, ones_like_actuals), "float"))
            same_actuals_count_op = tf.reduce_sum(tf.cast(tf.equal(actuals, twos_like_actuals), "float"))
            up_predictions_count_op = tf.reduce_sum(tf.cast(tf.equal(predictions, zeros_like_actuals), "float"))
            down_predictions_count_op = tf.reduce_sum(tf.cast(tf.equal(predictions, ones_like_actuals), "float"))
            same_predictions_count_op = tf.reduce_sum(tf.cast(tf.equal(predictions, twos_like_actuals), "float"))

            up_true_op = tf.reduce_sum(
                tf.cast(
                  tf.logical_and(
                    tf.equal(actuals, zeros_like_actuals),
                    tf.equal(predictions, zeros_like_predictions)
                  ),
                  "float"
                )
            )

            down_true_op = tf.reduce_sum(
                tf.cast(
                  tf.logical_and(
                    tf.equal(actuals, ones_like_actuals),
                    tf.equal(predictions, ones_like_predictions)
                  ),
                  "float"
                )
            )

            same_true_op = tf.reduce_sum(
                tf.cast(
                  tf.logical_and(
                    tf.equal(actuals, twos_like_actuals),
                    tf.equal(predictions, twos_like_predictions)
                  ),
                  "float"
                )
            )

            up_actuals_count, down_actuals_count, same_actuals_count, up_predictions_count, down_predictions_count, same_predictions_count = \
                sess.run(
                [up_actuals_count_op, down_actuals_count_op, same_actuals_count_op, up_predictions_count_op, down_predictions_count_op, same_predictions_count_op] ,
                feed_dict)

            up_true, down_true, same_true = \
                sess.run([up_true_op, down_true_op, same_true_op], feed_dict)

            up_pr = float(up_true) / float(up_predictions_count) if float(up_predictions_count) > 0 else 0
            down_pr = float(down_true) / float(down_predictions_count) if float(down_predictions_count) > 0 else 0
            same_pr = float(same_true) / float(same_predictions_count) if float(same_predictions_count) > 0 else 0

            accuracy = (float(up_true) + float(down_true) + float(same_true)) / \
                (float(up_predictions_count) + float(down_predictions_count) + float(same_predictions_count))

            print("Actual labels:")
            print("   UP  =", up_actuals_count)
            print("  DOWN =", down_actuals_count)
            print("  SAME =", same_actuals_count)
            print("\nPredicted labels:")
            print("   UP  =", up_predictions_count)
            print("  DOWN =", down_predictions_count)
            print("  SAME =", same_predictions_count)
            print("\nPrecision on each class:")
            print("   UP  =", up_pr)
            print("  DOWN =", down_pr)
            print("  SAME =", same_pr)
            print("\nAccuracy = ", accuracy)

    def backtest(self):
        tf.reset_default_graph()
        with tf.Session() as sess:
            feature_data = tf.placeholder("float", [None, self.num_predictors])
            labels = tf.placeholder("float", [None, self.num_classes])
            layers = [self.num_predictors] + self.hidden_layers + [self.num_classes]
            model = self.inference(feature_data, layers)
            feed_dict = {
                feature_data: self.test_predictors_tf.values,
                labels: self.test_classes_tf.values.reshape(len(self.test_classes_tf.values), self.num_classes)}

            # Restore parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.model_filename)

            predictions_op = tf.argmax(model, 1)
            actuals_op = tf.argmax(labels, 1)

            predictions, actuals = sess.run([predictions_op, actuals_op], feed_dict)
            pred2str = {0: "Up", 1: "Down", 2: "Same"}
            pred2int = {0: 1, 1: -1, 2: 0}
            pred2fee = {0: 525, 1: 525, 2: 0}

            df = pd.DataFrame(
                {"Prediction": [pred2str[s] for s in predictions],
                 "Actual": [pred2str[s] for s in actuals],
                 "Correct?": [s == actuals[i] for i, s in enumerate(predictions)],
                 "Open": self.raw_values["N225_open"].values,
                 "Close": self.raw_values["N225_close"].values,
                 "Diff": self.raw_values["N225_diff"].values,
                 "Profit/M": [pred2int[predictions[i]] * diff * 2 * 1000000 - (525 if predictions[i] != 2 else 0) for i, diff in enumerate(self.raw_values["N225_diff"].values)]},
                columns=["Prediction", "Actual", "Correct?", "Open", "Close", "Diff", "Profit/M"],
                index=self.raw_values["Date"].values)

            print(df.head())
            print(df.tail())

            print("Total Profit:", df.sum()["Profit/M"], "(" + str(df.sum()["Profit/M"] / 1000000 * 100) + "%)")

            df.to_csv(self.backtest_result)

    def backtest_monkey(self):
        N = len(self.raw_values["N225_diff"])

        def diff2class(x):
            if x >= self.threshold: return 0
            if x <= -self.threshold: return 1
            return 2


        total = []
        good = 0
        for step in range(100000):
            predictions = [random.randint(0, 2) for i in range(N)]
            actuals = [diff2class(x) for x in self.raw_values["N225_diff"].values]

            pred2str = {0: "Up", 1: "Down", 2: "Same"}
            pred2int = {0: 1, 1: -1, 2: 0}
            pred2fee = {0: 525, 1: 525, 2: 0}

            df = pd.DataFrame(
                {"Prediction": [pred2str[s] for s in predictions],
                 "Actual": [pred2str[s] for s in actuals],
                 "Correct?": [s == actuals[i] for i, s in enumerate(predictions)],
                 "Open": self.raw_values["N225_open"].values,
                 "Close": self.raw_values["N225_close"].values,
                 "Diff": self.raw_values["N225_diff"].values,
                 "Profit/M": [pred2int[predictions[i]] * diff * 2 * 1000000 - (525 if predictions[i] != 2 else 0) for i, diff in enumerate(self.raw_values["N225_diff"].values)]},
                columns=["Prediction", "Actual", "Correct?", "Open", "Close", "Diff", "Profit/M"],
                index=self.raw_values["Date"].values)

            total.append(df.sum()["Profit/M"])
            if total[-1] > 500000: good += 1

        ave = sum(total) / len(total)
        print("Average:", ave, "(" + str(ave / 1000000 * 100) + "%)")
        print("More than 50%:", good / 100000 * 100, "%", good)

    def predict(self, n225_open, today=np.datetime64("today"), downloadData=False):
        if downloadData:
            self.stockdata.download()

        start_date = today - np.timedelta64(14, 'D')
        end_date = today
        predictors_tf, _, raw_values = self.create_data(start_date=start_date, end_date=end_date)
        raw_values= raw_values["N225_close"][~np.isnan(raw_values["N225_close"])]

        # df = self.stockdata.get_data_for_prediction(today)
        # Get last values
        predictors_tf = predictors_tf[-2:-1]
        raw_values = raw_values[-1:]
        predictors_tf["N225_jump"] = (n225_open / raw_values.values[0]) - 1
        # print("Yesterday's N225 close value was", raw_values.values[0])
        # print("Will use this value for N225_jump:", predictors_tf["N225_jump"].values[0])

        tf.reset_default_graph()
        with tf.Session() as sess:
            feature_data = tf.placeholder("float", [None, self.num_predictors])
            layers = [self.num_predictors] + self.hidden_layers + [self.num_classes]
            model = self.inference(feature_data, layers)

            # Restore parameters
            saver = tf.train.Saver()
            saver.restore(sess, self.model_filename)

            prediction = sess.run(tf.argmax(model, 1), feed_dict={feature_data: predictors_tf.values})

            pred2str = {0: "Up", 1: "Down", 2: "Same"}
            print("\n")
            print("Today's N225:", pred2str[prediction[0]])
            print("\n")
