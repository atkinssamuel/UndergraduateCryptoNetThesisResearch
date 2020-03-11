import tensorflow as tf
import numpy as np
from base_classifier.project.consts import base_classifier_results_dir, base_classifier_checkpoint_dir


def test(x_test, y_test, checkpoint_file):
    # Parameters:
    # Base Params:
    input_nodes = x_test.shape[1]
    categories = 10
    image_dim = 28

    # Fully Connected:
    increase_factor = 1.5
    hidden_layer_1 = round(input_nodes * increase_factor)
    hidden_layer_2 = round(hidden_layer_1 * increase_factor)
    output_layer = 10

    # Defining Layers:
    # Defining Placeholders:
    x = tf.placeholder(tf.float32, [None, image_dim * image_dim])
    y_ = tf.placeholder(tf.float32, [None, categories])

    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_nodes, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = tf.math.square(tf.matmul(x, W1) + b1)
    # Layer 2 variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y2 = tf.math.square(tf.matmul(y1, W2) + b2)
    # Layer 3 Variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, output_layer], stddev=0.15))
    b3 = tf.Variable(tf.zeros([output_layer]))
    y = tf.nn.softmax(tf.matmul(y2, W3) + b3)

    cost = tf.reduce_sum(tf.math.square(y_ - y))
    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = base_classifier_checkpoint_dir + checkpoint_file

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        test_output = sess.run(y, feed_dict={x: x_test, y_: y_test})
        predictions = np.argmax(test_output, axis=1)
        targets = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(predictions, targets))/test_output.shape[0] * 100

        print(f"Testing Accuracy = {accuracy}%")

    return
