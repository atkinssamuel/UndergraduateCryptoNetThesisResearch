import tensorflow as tf
import numpy as np

from BostonHousingRegressionNets.Project.consts import *

if encrypted_flag:
    pass


def layer_width_investigation_train(x_train, y_train):
    # Parameters:
    # Base Params:
    input_dimension = x_train.shape[1]
    output_dimension = y_train.shape[1]

    hidden_layer_1 = 2048
    hidden_layer_2 = 1

    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_dimension])

    # Layer 1 variables:
    W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = tf.math.sigmoid(tf.matmul(x, W1) + b1)

    # Layer 2 variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y = tf.matmul(y1, W2) + b2

    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_dimension])

    cost = tf.reduce_sum(tf.math.square(y - y_))
    optimizer = tf.train.AdamOptimizer(TrainingParameters.learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=TrainingParameters.num_models)

    training_losses = []

    start_time = time.time()

    with tf.Session(config=EncryptionParameters.config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(TrainingParameters.num_epochs):
            for batch in range(int(sample_count / TrainingParameters.batch_size)):
                batch_x = x_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                batch_y = y_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                # Instantiating the inputs and targets with the batch values:
                optim_out = np.array(sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y}))
            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"{round(epoch_iteration / TrainingParameters.num_epochs * 100, 2)}% Complete, "
                  f"Time Elapsed = {round(time.time() - start_time, 3)}s")

            if epoch_iteration % TrainingParameters.checkpoint_frequency == 0:
                np.save(training_results_numpy_save_dir +
                        f"{TrainingParameters.training_output_numpy_file_path}{epoch_iteration}", training_output)
                np.save(training_results_numpy_save_dir +
                        f"{TrainingParameters.training_targets_numpy_file_path}{epoch_iteration}", y_train)
                checkpoint = f"{TrainingParameters.incomplete_checkpoint_file_location}{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)
        sess.close()
    with open(results_dir + "MagVsLayerWidthSigmoid/" + f'training_output_mean_{hidden_layer_1}.csv', 'a') \
            as fd:
        fd.write(f"{np.abs(np.mean(training_output))}, ")
    with open(results_dir + "MagVsLayerWidthSigmoid/" + f'initial_training_loss_{hidden_layer_1}.csv', 'a') \
            as fd:
        fd.write(f"{training_losses[0]}, ")
    training_losses = np.array(training_losses)
    np.savetxt(training_results_numpy_save_dir + f"training_losses.csv", training_losses, delimiter=',')
    np.save(training_results_numpy_save_dir + f"training_losses", training_losses)
    max_value = np.max(training_losses)
    for i in range(np.size(training_losses)):
        if i % TrainingParameters.checkpoint_frequency != 0:
            training_losses[i] = max_value
    min_index = np.argmin(training_losses)
    time_elapsed = round(time.time() - start_time, 3)

    print(f"Total Training Time Elapsed = {time_elapsed}s")
    print(f"Min Training Loss = {training_losses[min_index]} at Epoch {min_index}")

    np.save(training_results_save_dir + "training_time_elapsed", time_elapsed)
    return
