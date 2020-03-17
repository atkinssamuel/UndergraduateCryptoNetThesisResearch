import time
import numpy as np
import tensorflow as tf

from PerformanceComparison.Regression.PlaintextModel.Project.consts import *
from PerformanceComparison.Regression.PlaintextModel.Project.helpers import plot_regression_outputs


def test(x_test, y_test):
    # Parameters:
    # Base Params:
    input_dimension = x_test.shape[1]
    output_dimension = y_test.shape[1]
    complexity_scaling_factor = 3.5

    # Defining Placeholders:
    x = tf.placeholder(tf.float32, [None, input_dimension])
    y_ = tf.placeholder(tf.float32, [None, output_dimension])

    # Layer 1a: Fully Connected
    l1_output_nodes = round(input_dimension * complexity_scaling_factor)
    W1 = tf.Variable(tf.truncated_normal([input_dimension, l1_output_nodes]))
    b1 = tf.Variable(tf.zeros([l1_output_nodes]))
    y1 = tf.matmul(x, W1) + b1

    # Layer 1b: Sigmoid Activation
    y1 = tf.math.sigmoid(y1)

    # Layer 2a: Fully Connected
    l2_output_nodes = round(l1_output_nodes * complexity_scaling_factor)
    W2 = tf.Variable(tf.truncated_normal([l1_output_nodes, l2_output_nodes]))
    b2 = tf.Variable(tf.zeros([l2_output_nodes]))
    y2 = tf.matmul(y1, W2) + b2

    # Layer 2b: Sigmoid Activation
    y2 = tf.math.sigmoid(y2)

    # Layer 3a: Fully Connected
    W3 = tf.Variable(tf.truncated_normal([l2_output_nodes, output_dimension]))
    b3 = tf.Variable(tf.zeros([output_dimension]))
    y = tf.matmul(y2, W3) + b3

    cost = tf.reduce_sum(tf.math.square(y_ - y))


    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = TestingParameters.checkpoint_file_location

    start_time = time.time()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        test_output, test_loss = sess.run([y, cost], feed_dict={x: x_test, y_: y_test})

        plot_regression_outputs(test_output, y_test, title="Testing Output Comparison:", x_label="Data Index",
                                y_label="Housing Price in USD", save_directory=testing_results_save_dir,
                                save_title="testing_output.png")

    time_elapsed = round(time.time() - start_time, 3)
    print(f"Testing Time Elapsed = {time_elapsed}s")
    print(f"Test Loss = {test_loss}")
    np.save(testing_results_save_dir + "testing_time_elapsed", time_elapsed)
    np.save(testing_results_save_dir + "testing_loss", test_loss)
    return
