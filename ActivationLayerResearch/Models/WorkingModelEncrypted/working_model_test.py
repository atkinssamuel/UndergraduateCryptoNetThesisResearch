import numpy as np
import tensorflow as tf

from Project.consts import *

def working_model_encrypted_test(x_test, y_test):
    # Parameters:
    input_dimension = x_test.shape[1]
    output_dimension = y_test.shape[1]
    complexity_scaling_factor = 1.5

    hidden_layer_1 = 64
    hidden_layer_2 = 1

    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_dimension])

    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = tf.math.square(tf.matmul(x, W1) + b1)

    # layer 2 variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y = tf.matmul(y1, W2) + b2

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
    np.save(testing_results_numpy_save_dir + TestingParameters.testing_output_numpy_file_path, test_output)
    np.save(testing_results_numpy_save_dir + TestingParameters.testing_targets_numpy_file_path, y_test)
    np.save(testing_results_save_dir + "testing_time_elapsed", time_elapsed)
    np.save(testing_results_save_dir + "testing_loss", test_loss)
    return
