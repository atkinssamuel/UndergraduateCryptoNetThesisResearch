import numpy as np
import tensorflow as tf

from BostonHousingRegressionNets.Project.consts import *

def scaled_squared_test(x_test, y_test):
    print("scaled_squared_test")
    # Parameters:
    input_dimension = x_test.shape[1]
    output_dimension = y_test.shape[1]

    layer_complexity_growth = 2
    l1_scaling, l2_scaling, l3_scaling = 0.001, 0.001, 0.001
    hidden_layer_1 = 32
    hidden_layer_2 = hidden_layer_1 * layer_complexity_growth
    hidden_layer_3 = hidden_layer_2 * layer_complexity_growth
    output_layer = 1

    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_dimension])

    # Layer 1 Variables:
    W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = l1_scaling * tf.math.square(tf.matmul(x, W1) + b1)

    # Layer 2 Variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y2 = l2_scaling * tf.math.square(tf.matmul(y1, W2) + b2)

    # Layer 3 Variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, hidden_layer_3], stddev=0.15))
    b3 = tf.Variable(tf.zeros([hidden_layer_3]))
    y3 = l3_scaling * tf.math.square(tf.matmul(y2, W3) + b3)

    # Output Layer Variables:
    W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, output_layer], stddev=0.15))
    b4 = tf.Variable(tf.zeros([output_layer]))
    y = tf.matmul(y3, W4) + b4

    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_dimension])

    cost = tf.reduce_sum(tf.math.square(y - y_))
    optimizer = tf.train.AdamOptimizer(TrainingParameters.learning_rate).minimize(cost)


    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = TestingParameters.checkpoint_file_location

    start_time = time.time()

    with tf.Session(config=EncryptionParameters.config) as sess:
        saver.restore(sess, checkpoint)
        test_output, test_loss = sess.run([y, cost], feed_dict={x: x_test, y_: y_test})



    time_elapsed = round(time.time() - start_time, 3)
    print(f"Testing Time Elapsed = {time_elapsed}s")
    print(f"Test Loss = {test_loss}")
    np.save(testing_results_numpy_save_dir + TestingParameters.testing_numpy_file_path, test_output)
    np.save(testing_results_numpy_save_dir + TestingParameters.testing_targets_numpy_file_path, y_test)
    np.save(testing_results_save_dir + "testing_time_elapsed", time_elapsed)
    np.save(testing_results_save_dir + "testing_loss", test_loss)

    with open(results_dir + "Performance/" + f'testing_performance.txt', 'a') \
            as fd:
        fd.write(f"Testing Loss = {test_loss}, Time Elapsed = {time_elapsed}\n")
    return
