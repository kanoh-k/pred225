import pandas as pd
import tensorflow as tf
from datetime import datetime
from stockdata import StockData, INDEX_LIST, INDEX_TO_DISPLAY_TEXT, CURRENCY_PAIR_LIST

def inference(feature_data, layers):
    """ Create a model with given numbers of layters.

    'layers' is the list of integers. Each element in the list defines the number
    of nodes in each layer. layers[0] is an input layer, layers[-1] is an output
    layer. Other layers[i] means i-th hidden layer.
    :type feature_data: ???
    :type layers: List<int>
    """
    if len(layers) < 2:
        raise Exception("'layers' should have more than one elements")

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

# Returns loss function and its summary string
def loss(model, labels):
    cost = -tf.reduce_sum(labels*tf.log(model))
    return cost ,tf.scalar_summary("Cross entropy", cost)

def training(cost, learning_rate=0.0001):
    training_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return training_op

# Note: This function is currently invalid as # of output classes
#       has been changed to three. I will re-implement this later.
def tf_confusion_metrics(model, labels, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(labels, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals),
        tf.equal(predictions, ones_like_predictions)
      ),
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals),
        tf.equal(predictions, zeros_like_predictions)
      ),
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op],
      feed_dict
    )

  tpr = float(tp)/(float(tp) + float(fn))
  fpr = float(fp)/(float(tp) + float(fn))

  accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

  recall = tpr
  precision = float(tp)/(float(tp) + float(fp))

  f1_score = (2 * (precision * recall)) / (precision + recall)

  print ('Precision = ', precision)
  print ('Recall = ', recall)
  print ('F1 Score = ', f1_score)
  print ('Accuracy = ', accuracy)

def create_training_test_data():
    s = StockData()
    #s.download()
    opening_data, closing_data, diff_data, jump_data = s.get_daily_data(1000, True, True)

    threshold = 0.003
    closing_data["N225_up"] = 0
    closing_data.ix[diff_data["N225"] >= threshold, "N225_up"] = 1
    closing_data["N225_down"] = 0
    closing_data.ix[diff_data["N225"] <= -threshold, "N225_down"] = 1
    closing_data["N225_same"] = 0
    closing_data.ix[(-threshold < diff_data["N225"]) & (diff_data["N225"] < threshold), "N225_same"] = 1

    columns = ["N225_up", "N225_down", "N225_same"] + ["N225_jump"]\
    + [s + "_1_o" for s in INDEX_LIST[1:]] + [s + "_1_o" for s in CURRENCY_PAIR_LIST] \
    + [s + "_1_c" for s in INDEX_LIST[1:]] + [s + "_1_c" for s in CURRENCY_PAIR_LIST] \
    + [s + "_1_d" for s in INDEX_LIST[1:]] + [s + "_1_d" for s in CURRENCY_PAIR_LIST]

    training_test_data = pd.DataFrame(
        # column name is "<index>_<day>_<type>".
        # E.g., "DJI_1_c" means yesterday's Dow closing value.
        columns=columns
    )

    for i in range(7, len(closing_data)):
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
        training_test_data = training_test_data.append(data, ignore_index=True)

    # Prepare training data and test data
    predictors_tf = training_test_data[training_test_data.columns[3:]]
    classes_tf = training_test_data[training_test_data.columns[:3]]

    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size

    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]
    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]
    return training_predictors_tf, training_classes_tf, test_predictors_tf, test_classes_tf

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
        summary_writer = tf.train.SummaryWriter("/tmp/pred225_log2/" + experiment_name, sess.graph)

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

        tf_confusion_metrics(model, labels, sess, test_dict)

def evaluate_models():
    experiments = [
        [],
        [25],
        [50],
        [100],
        [50, 25],
        [100, 50],
        [50, 25, 10],
        [100, 50, 25],
        [200, 100, 50]
    ]

    for layers in experiments:
        evaluate_model(layers)

if __name__ == "__main__":
    evaluate_models()
