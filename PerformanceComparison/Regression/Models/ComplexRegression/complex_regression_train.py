import tensorflow as tf
import numpy as np

from Project.consts import *

if encrypted_flag:
    import ngraph_bridge


def complex_regression_train(x_train, y_train, x_valid, y_valid):
    print("complex_regression_train")
    # Parameters:
    # Base Params:
    input_dimension = x_train.shape[1]
    output_dimension = y_train.shape[1]

    layer_complexity_growth = 2
    l1_scaling, l2_scaling, l3_scaling = 1, 1, 1
    hidden_layer_1 = input_dimension
    hidden_layer_2 = hidden_layer_1 * layer_complexity_growth
    hidden_layer_3 = hidden_layer_2 * layer_complexity_growth
    output_layer = 1

    # Placeholder for batch of inputs:
    x = tf.placeholder(tf.float32, [None, input_dimension])

    # Layer 1 Variables:
    W1 = tf.Variable(tf.truncated_normal([input_dimension, hidden_layer_1], stddev=0.15))
    b1 = tf.Variable(tf.zeros([hidden_layer_1]))
    y1 = l1_scaling * tf.math.sigmoid(tf.matmul(x, W1) + b1)

    # Layer 2 Variables:
    W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, hidden_layer_2], stddev=0.15))
    b2 = tf.Variable(tf.zeros([hidden_layer_2]))
    y2 = l2_scaling * tf.math.sigmoid(tf.matmul(y1, W2) + b2)

    # Layer 3 Variables:
    W3 = tf.Variable(tf.truncated_normal([hidden_layer_2, hidden_layer_3], stddev=0.15))
    b3 = tf.Variable(tf.zeros([hidden_layer_3]))
    y3 = l3_scaling * tf.math.sigmoid(tf.matmul(y2, W3) + b3)

    # Output Layer Variables:
    W4 = tf.Variable(tf.truncated_normal([hidden_layer_3, output_layer], stddev=0.15))
    b4 = tf.Variable(tf.zeros([output_layer]))
    y = tf.matmul(y3, W4) + b4

    # Placeholder for batch of targets:
    y_ = tf.placeholder(tf.float32, [None, output_dimension])

    cost = tf.reduce_sum(tf.math.square(y - y_))
    optimizer = tf.train.AdamOptimizer(TrainingParameters.learning_rate).minimize(cost)

    # Miscellaneous quantities:
    sample_count = np.shape(x_train)[0]

    # For weight saving:
    saver = tf.train.Saver(max_to_keep=TrainingParameters.num_models)

    training_losses = []
    validation_losses = []

    start_time = time.time()

    with tf.Session(config=EncryptionParameters.config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_iteration in range(TrainingParameters.num_epochs):
            for batch in range(int(sample_count / TrainingParameters.batch_size)):
                batch_x = x_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                batch_y = y_train[batch * TrainingParameters.batch_size: (1 + batch) * TrainingParameters.batch_size]
                # Instantiating the inputs and targets with the batch values:
                optim_out = np.array(sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y}))

            # Appending loss and validation loss to appropriate arrays:
            training_output, training_loss = sess.run([y, cost], feed_dict={x: x_train, y_: y_train})
            validation_loss = sess.run([cost], feed_dict={x: x_valid, y_: y_valid})
            validation_losses.append(validation_loss)
            training_loss = np.mean(training_loss)
            training_losses.append(training_loss)

            # Printing status update:
            print(f"Current Epoch = {epoch_iteration}, Training Loss = {training_loss}, "
                  f"{round(epoch_iteration / TrainingParameters.num_epochs * 100, 2)}% Complete, "
                  f"Time Elapsed = {round(time.time() - start_time, 3)}s")

            # Checkpoint:
            if epoch_iteration % TrainingParameters.checkpoint_frequency == 0:
                np.save(training_results_numpy_save_dir +
                        f"{TrainingParameters.training_output_numpy_file_path}{epoch_iteration}", training_output)
                np.save(training_results_numpy_save_dir +
                        f"{TrainingParameters.training_targets_numpy_file_path}{epoch_iteration}", y_train)
                checkpoint = f"{TrainingParameters.incomplete_checkpoint_file_location}{epoch_iteration}.ckpt"
                saver.save(sess, checkpoint)

        sess.close()
    # Saving training losses:
    training_losses = np.array(training_losses)
    np.savetxt(training_results_numpy_save_dir + f"{TrainingParameters.training_losses_numpy_file_path}.csv",
               training_losses, delimiter=',')
    np.save(training_results_numpy_save_dir + f"{TrainingParameters.training_losses_numpy_file_path}", training_losses)
    # Saving validation losses:
    validation_losses = np.array(validation_losses)
    np.savetxt(training_results_numpy_save_dir + f"{TrainingParameters.validation_losses_numpy_file_path}.csv",
               validation_losses, delimiter=',')
    np.save(training_results_numpy_save_dir + f"{TrainingParameters.validation_losses_numpy_file_path}",
            validation_losses)
    max_value = np.max(validation_losses)
    for i in range(np.size(validation_losses)):
        if i % TrainingParameters.checkpoint_frequency != 0:
            validation_losses[i] = max_value
    min_index = np.argmin(validation_losses)
    time_elapsed = round(time.time() - start_time, 3)

    print(f"Total Training Time Elapsed = {time_elapsed}s")
    print(f"Min Validation Loss = {validation_losses[min_index][0]} at Epoch {min_index}")

    np.save(training_results_save_dir + "training_time_elapsed", time_elapsed)
    np.savetxt(training_results_save_dir + "training_time_elapsed.csv", [time_elapsed], delimiter=',')
    return


