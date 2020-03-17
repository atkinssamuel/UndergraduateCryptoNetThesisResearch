import time
import numpy as np
import tensorflow as tf

from mnist_models.microsoft_cryptonet_model.project.consts import checkpoint_dir


def test(x_test, y_test, checkpoint_file):
    # Parameters:
    # Base Params:
    input_depth = 1
    categories = 10
    image_dim = 28

    # Defining Placeholders:
    x = tf.placeholder(tf.float32, [None, image_dim, image_dim, input_depth])
    y_ = tf.placeholder(tf.float32, [None, categories])


    # Layer 1: Convolutional
    l1_padding = tf.constant([[0, 0], [1, 0], [1, 0], [0, 0]])
    l1 = tf.pad(x, l1_padding, "CONSTANT")
    k1_width = 5
    l1_feature_maps = 5
    l1_padding_type = "VALID"
    l1_strides = [2, 2]
    k1 = tf.Variable(tf.random_normal([k1_width, k1_width, input_depth, l1_feature_maps]))
    y1 = tf.nn.conv2d(l1, filter=k1, strides=l1_strides, padding=l1_padding_type)

    # Layer 2: Square Activation
    y2 = tf.square(y1)

    # Layer 3: Scaled Mean Pool
    l3_window_shape = [3, 3]
    l3_strides = [1, 1]
    y3 = tf.nn.pool(y2, window_shape=l3_window_shape, pooling_type="AVG", padding="SAME", strides=l3_strides)

    # Layer 4: Convolutional
    k4_width = 5
    l4_feature_maps = 50
    l4_padding_type = "VALID"
    l4_strides = [2, 2]
    k4 = tf.Variable(tf.random_normal([k4_width, k4_width, l1_feature_maps, l4_feature_maps]))
    y4 = tf.nn.conv2d(y3, filter=k4, strides=l4_strides, padding=l4_padding_type)

    # Layer 5: Scaled Mean Pool:
    l5_window_shape = [3, 3]
    l5_strides = [1, 1]
    y5 = tf.nn.pool(y4, window_shape=l5_window_shape, pooling_type="AVG", padding="SAME", strides=l5_strides)
    l5_output_dim = 5
    y5 = tf.reshape(y5, (-1, l5_output_dim * l5_output_dim * l4_feature_maps))

    # Layer 6: Fully Connected
    l6_output_nodes = 100
    W6 = tf.Variable(tf.truncated_normal([l5_output_dim * l5_output_dim * l4_feature_maps, l6_output_nodes]))
    b6 = tf.Variable(tf.zeros([l6_output_nodes]))
    y6 = tf.matmul(y5, W6) + b6

    # Layer 7: Square Activation
    y7 = tf.square(y6)

    # Layer 8: Fully Connected
    l8_output_nodes = 10
    W8 = tf.Variable(tf.truncated_normal([l6_output_nodes, l8_output_nodes]))
    b8 = tf.Variable(tf.zeros(l8_output_nodes))
    y = tf.matmul(y7, W8) + b8

    # For weight saving:
    saver = tf.train.Saver()
    checkpoint = checkpoint_dir + checkpoint_file

    start_time = time.time()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)

        test_output = sess.run(y, feed_dict={x: x_test, y_: y_test})
        predictions = np.argmax(test_output, axis=1)
        targets = np.argmax(y_test, axis=1)
        accuracy = np.sum(np.equal(predictions, targets))/test_output.shape[0] * 100

        print(f"Testing Accuracy = {accuracy}%")
    print(f"Testing Time Elapsed = {round(time.time() - start_time, 3)}s")
    return
