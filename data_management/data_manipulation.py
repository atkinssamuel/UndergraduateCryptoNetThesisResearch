from mnist import MNIST
import numpy as np

MNIST_dir = "../data_management/dataset/"
np_save_dir = "../data_management/np_dataset/"


def import_MNIST():
    # Ensure you replace the dots in the dataset files with "-"s to avoid a "File not found" error
    MNIST_data = MNIST(MNIST_dir)

    x_train, y_train = MNIST_data.load_training()
    x_test, y_test = MNIST_data.load_testing()

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Encoding outputs:
    y_train_categorical = np.zeros((y_train.shape[0], 10))
    y_test_categorical = np.zeros((y_test.shape[0], 10))

    for i in range(y_train_categorical.shape[0]):
        j = y_train[i]
        y_train_categorical[i, j] = 1
        if i < y_test_categorical.shape[0]:
            k = y_test[i]
            y_test_categorical[i, k] = 1

    np.save(np_save_dir + 'x_train', x_train)
    np.save(np_save_dir + 'y_train', y_train_categorical)
    np.save(np_save_dir + 'x_test', x_test)
    np.save(np_save_dir + 'y_test', y_test_categorical)
    return

