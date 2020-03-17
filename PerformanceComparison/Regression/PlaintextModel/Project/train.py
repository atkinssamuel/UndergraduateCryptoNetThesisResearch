import tensorflow as tf
import numpy as np
import time

from PerformanceComparison.Regression.PlaintextModel.Project.consts import *
from PerformanceComparison.Regression.PlaintextModel.Project.helpers import plot_regression_outputs, plot_loss


def train(x_train, y_train):
    # Parameters:
    # Base Params:
    input_dimension = x_train.shape[1]
    output_dimension = y_train.shape[1]
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
    # cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

    optimizer = tf.train.AdamOptimizer(TrainingParameters.learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=TrainingParameters.num_models)

    training_losses = []

    start_time = time.time()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(TrainingParameters.num_epochs):
            for batch in range(int(sample_count / TrainingParameters.batch_size)):
                batch_x = x_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                batch_y = y_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                # Instantiating the inputs and targets with the batch values:
                output = np.array(sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y}))

            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"{round(epoch_iteration / TrainingParameters.num_epochs * 100, 2)}% Complete, "
                  f"Time Elapsed = {round(time.time() - start_time, 3)}s")

            if epoch_iteration % TrainingParameters.checkpoint_frequency == 0:
                plot_regression_outputs(training_output, y_train, title=f"Training Output Comparison at Epoch {epoch_iteration}:",
                                        x_label="Data Index", y_label= "Housing Price in USD",
                                        save_directory=training_results_save_dir,
                                        save_title=f"training_output_comparison_epoch_{epoch_iteration}.png",
                                        show=False)
                checkpoint = f"{TrainingParameters.incomplete_checkpoint_file_location}{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()

    time_elapsed = round(time.time() - start_time, 3)
    print(f"Total Training Time Elapsed = {time_elapsed}s")

    plot_loss(training_losses, title=f"Training Losses:", x_label="Epoch Iteration", y_label="Loss",
              save_directory=training_results_save_dir, save_title="training_loss.png",
              show=True)
    np.save(training_results_save_dir + "training_time_elapsed", time_elapsed)
    return
