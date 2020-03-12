import tensorflow as tf
import numpy as np
from mnist_models.microsoft_cryptonet_model.project.consts import *

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


    # Layer 1: Convolutional:
    l1_paddings = tf.constant([[0, 0], [1, 0], [0, 1], [0, 0]])
    l1 = tf.pad(x, l1_paddings, "CONSTANT")
    k1_width = 5
    l1_feature_maps = 5
    l1_padding = "VALID"
    k1 = tf.Variable(tf.random_normal([k1_width, k1_width, input_depth, l1_feature_maps]))
    y1 = tf.nn.conv2d(l1, filter=k1, strides=[2, 2], padding=l1_padding)


    y = y1

    # cost = tf.reduce_sum(tf.math.square(y_ - y))
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=num_models)

    training_losses = []
    training_accuracies = []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(num_epochs):
            for batch in range(int(sample_count / batch_size)):
                batch_x = x_train[batch * batch_size: (1 + batch) * batch_size]
                batch_y = y_train[batch * batch_size: (1 + batch) * batch_size]
                # Instantiating the inputs and targets with the batch values:
                output = np.array(sess.run([y1], feed_dict={x: batch_x, y_: batch_y}))
                print(output.shape)
            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)
            training_predictions = np.argmax(training_output, axis=1)
            training_targets = np.argmax(y_train, axis=1)
            training_accuracy = round(np.sum(np.equal(training_predictions, training_targets)) \
                                / training_predictions.shape[0] * 100, 2)
            training_accuracies.append(training_accuracy)
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"Training Accuracy = {training_accuracy}%, {round(epoch_iteration / num_epochs * 100, 2)}% Complete")

            if epoch_iteration % checkpoint_frequency == 0:
                checkpoint = checkpoint_dir + f"conv_epoch_{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()
    if encrypted_flag:
        np.save(encrypted_numpy_dir + 'encrypted_training_losses', training_losses)
        np.save(encrypted_numpy_dir + 'encrypted_training_accuracies', training_accuracies)
    else:
        np.save(unencrypted_numpy_dir + 'unencrypted_training_losses', training_losses)
        np.save(unencrypted_numpy_dir + 'unencrypted_training_accuracies', training_accuracies)

    return
