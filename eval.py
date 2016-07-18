import pandas as pd
import tensorflow as tf
from stockdata import StockData, INDEX_LIST, INDEX_TO_DISPLAY_TEXT, CURRENCY_PAIR_LIST

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

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
    closing_data = s.get_closing_data(1000, True, True)

    closing_data["N225_positive"] = 0
    closing_data.ix[closing_data["N225"] >= 0, "N225_positive"] = 1
    closing_data["N225_negative"] = 0
    closing_data.ix[closing_data["N225"] < 0, "N225_negative"] = 1

    training_test_data = pd.DataFrame(
        # column name is "<index>_<day>".
        # E.g., "DJI_1" means yesterday's Dow.
        columns= ["N225_positive", "N225_negative"] + [s + "_1" for s in INDEX_LIST[1:]] + [s + "_1" for s in CURRENCY_PAIR_LIST]
    )

    for i in range(7, len(closing_data)):
        data = {}
        # We will use today's data for positive/negative labels
        data["N225_positive"] = closing_data["N225_positive"].ix[i]
        data["N225_negative"] = closing_data["N225_negative"].ix[i]
        # Use yesterday's data for world market data
        for col in INDEX_LIST[1:] + CURRENCY_PAIR_LIST:
            data[col + "_1"] = closing_data[col].ix[i - 1]
        training_test_data = training_test_data.append(data, ignore_index=True)

    # Prepare training data and test data
    predictors_tf = training_test_data[training_test_data.columns[2:]]
    classes_tf = training_test_data[training_test_data.columns[:2]]

    training_set_size = int(len(training_test_data) * 0.8)
    test_set_size = len(training_test_data) - training_set_size

    training_predictors_tf = predictors_tf[:training_set_size]
    training_classes_tf = classes_tf[:training_set_size]
    test_predictors_tf = predictors_tf[training_set_size:]
    test_classes_tf = classes_tf[training_set_size:]
    return training_predictors_tf, training_classes_tf, test_predictors_tf, test_classes_tf

# 2 hideen layers
def evaluate_model():
    training_predictors_tf, training_classes_tf, test_predictors_tf, test_classes_tf = create_training_test_data()
    sess1 = tf.Session()

    num_predictors = len(training_predictors_tf.columns)
    num_classes = len(training_classes_tf.columns)
    num_indices = training_predictors_tf.shape[1]

    feature_data = tf.placeholder("float", [None, num_predictors])
    actual_classes = tf.placeholder("float", [None, 2])

    weights1 = tf.Variable(tf.truncated_normal([num_indices, 50], stddev=0.0001))
    biases1 = tf.Variable(tf.ones([50]))

    weights2 = tf.Variable(tf.truncated_normal([50, 25], stddev=0.0001))
    biases2 = tf.Variable(tf.ones([25]))

    weights3 = tf.Variable(tf.truncated_normal([25, 2], stddev=0.0001))
    biases3 = tf.Variable(tf.ones([2]))

    # This time we introduce a single hidden layer into our model...
    hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
    model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

    cost = -tf.reduce_sum(actual_classes*tf.log(model))

    train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

    init = tf.initialize_all_variables()
    sess1.run(init)

    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    for i in range(1, 30001):
      sess1.run(
        train_op1,
        feed_dict={
          feature_data: training_predictors_tf.values,
          actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
        }
      )
      if i%5000 == 0:
        print (i, sess1.run(
          accuracy,
          feed_dict={
            feature_data: training_predictors_tf.values,
            actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
          }
        ))

    feed_dict= {
        feature_data: test_predictors_tf.values,
        actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)}

    tf_confusion_metrics(model, actual_classes, sess1, feed_dict)

evaluate_model()