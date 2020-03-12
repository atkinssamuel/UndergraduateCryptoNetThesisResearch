import numpy as np

np_save_dir = "../data_management/np_dataset/"


def load_MNIST(train_percentage=100, test_percentage=100):
    image_dim = 28
    # num_categories = 10

    x_train = np.load(np_save_dir + "x_train.npy")
    y_train = np.load(np_save_dir + "y_train.npy")
    x_test = np.load(np_save_dir + "x_test.npy")
    y_test = np.load(np_save_dir + "y_test.npy")

    x_train = x_train.reshape((np.shape(x_train)[0], image_dim, image_dim, 1))
    x_test = x_test.reshape((np.shape(x_test)[0], image_dim, image_dim, 1))

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage/100 * train_count)
    adjusted_test_count = round(test_percentage/100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_MNIST_flat(train_percentage=100, test_percentage=100):
    x_train = np.load(np_save_dir + "x_train.npy")
    y_train = np.load(np_save_dir + "y_train.npy")
    x_test = np.load(np_save_dir + "x_test.npy")
    y_test = np.load(np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage/100 * train_count)
    adjusted_test_count = round(test_percentage/100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)
