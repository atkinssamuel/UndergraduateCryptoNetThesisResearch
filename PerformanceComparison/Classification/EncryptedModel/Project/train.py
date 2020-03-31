import tensorflow as tf
import numpy as np
from PerformanceComparison.Classification.EncryptedModel.Project.consts import *
import time

if encrypted_flag:
    import ngraph_bridge


def train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=10, num_models=200, config=None):
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
    l1_scaling = 0.01
    y2 = l1_scaling * tf.square(y1)

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
    l7_scaling = 0.01
    y7 = l7_scaling * tf.square(y6)

    # Layer 8: Fully Connected
    l8_output_nodes = 10
    W8 = tf.Variable(tf.truncated_normal([l6_output_nodes, l8_output_nodes]))
    b8 = tf.Variable(tf.zeros(l8_output_nodes))
    y8 = tf.matmul(y7, W8) + b8

    # Layer 9: Sigmoid Activation
    # Sigmoid is implicit in the cost function
    y = tf.math.sigmoid(y8)

    # cost = tf.reduce_sum(tf.math.square(y_ - y))
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=num_models)

    training_losses = []
    training_accuracies = []

    start_time = time.time()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count / batch_size)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                # Instantiating the inputs and targets with the batch values:
                output = np.array(sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y}))

            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)
            training_predictions = np.argmax(training_output, axis=1)
            training_targets = np.argmax(y_train, axis=1)
            training_accuracy = round(np.sum(np.equal(training_predictions, training_targets)) \
                                / training_predictions.shape[0] * 100, 2)
            training_accuracies.append(training_accuracy)
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"Training Accuracy = {training_accuracy}%, {round(epoch_iteration / num_epochs * 100, 2)}% Complete, "
                  f"Time Elapsed = {round(time.time() - start_time, 3)}s")

            if epoch_iteration % checkpoint_frequency == 0:
                checkpoint = f"{TrainingParameters.incomplete_checkpoint_file_location}{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()
    np.save(training_results_numpy_dir + 'training_losses', training_losses)
    np.save(training_results_numpy_dir + 'training_accuracies', training_accuracies)

    print(f"Total Training Time Elapsed = {round(time.time() - start_time, 3)}s")
    return
