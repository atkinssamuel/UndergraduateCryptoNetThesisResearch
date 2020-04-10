import numpy as np
from DataManagement.data_constants import *


def load_MNIST(train_percentage=100, test_percentage=100):
    image_dim = 28
    # num_categories = 10

    x_train = np.load(MNIST_constants.mnist_np_save_dir + "x_train.npy")
    y_train = np.load(MNIST_constants.mnist_np_save_dir + "y_train.npy")
    x_test = np.load(MNIST_constants.mnist_np_save_dir + "x_test.npy")
    y_test = np.load(MNIST_constants.mnist_np_save_dir + "y_test.npy")

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
    x_train = np.load(MNIST_constants.mnist_np_save_dir + "x_train.npy")
    y_train = np.load(MNIST_constants.mnist_np_save_dir + "y_train.npy")
    x_test = np.load(MNIST_constants.mnist_np_save_dir + "x_test.npy")
    y_test = np.load(MNIST_constants.mnist_np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage/100 * train_count)
    adjusted_test_count = round(test_percentage/100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_cifar100(train_percentage=100, test_percentage=100):
    x_train = np.load(cifar100_constants.cifar100_np_save_dir + "x_train.npy")
    y_train = np.load(cifar100_constants.cifar100_np_save_dir + "y_train.npy")
    x_test = np.load(cifar100_constants.cifar100_np_save_dir + "x_test.npy")
    y_test = np.load(cifar100_constants.cifar100_np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage/100 * train_count)
    adjusted_test_count = round(test_percentage/100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_cifar10(train_percentage=100, test_percentage=100):
    x_train = np.load(cifar10_constants.cifar10_np_save_dir + "x_train.npy")
    y_train = np.load(cifar10_constants.cifar10_np_save_dir + "y_train.npy")
    x_test = np.load(cifar10_constants.cifar10_np_save_dir + "x_test.npy")
    y_test = np.load(cifar10_constants.cifar10_np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage / 100 * train_count)
    adjusted_test_count = round(test_percentage / 100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_fashion_mnist(train_percentage=100, test_percentage=100):
    x_train = np.load(fashion_mnist_constants.fashion_mnist_np_save_dir + "x_train.npy")
    y_train = np.load(fashion_mnist_constants.fashion_mnist_np_save_dir + "y_train.npy")
    x_test = np.load(fashion_mnist_constants.fashion_mnist_np_save_dir + "x_test.npy")
    y_test = np.load(fashion_mnist_constants.fashion_mnist_np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage / 100 * train_count)
    adjusted_test_count = round(test_percentage / 100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_boston_housing(train_percentage=100, test_percentage=100):
    x_train = np.load(boston_housing_constants.boston_housing_np_save_dir + "x_train.npy")
    y_train = np.load(boston_housing_constants.boston_housing_np_save_dir + "y_train.npy")
    x_test = np.load(boston_housing_constants.boston_housing_np_save_dir + "x_test.npy")
    y_test = np.load(boston_housing_constants.boston_housing_np_save_dir + "y_test.npy")

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_test = y_test.reshape((y_test.shape[0], 1))

    train_count = x_train.shape[0]
    test_count = x_test.shape[0]
    adjusted_train_count = round(train_percentage / 100 * train_count)
    adjusted_test_count = round(test_percentage / 100 * test_count)

    (x_train, y_train), (x_test, y_test) = (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
                                           (x_test[:adjusted_test_count], y_test[:adjusted_test_count])

    return (x_train, y_train), (x_test, y_test)


def load_year_prediction(train_percentage=100, validation_percentage=100, test_percentage=100):
    x_train = np.load(year_prediction_constants.year_prediction_np_save_dir + "x_train.npy")
    y_train = np.load(year_prediction_constants.year_prediction_np_save_dir + "y_train.npy")
    x_valid = np.load(year_prediction_constants.year_prediction_np_save_dir + "x_valid.npy")
    y_valid = np.load(year_prediction_constants.year_prediction_np_save_dir + "y_valid.npy")
    x_test = np.load(year_prediction_constants.year_prediction_np_save_dir + "x_test.npy")
    y_test = np.load(year_prediction_constants.year_prediction_np_save_dir + "y_test.npy")

    train_count = x_train.shape[0]
    valid_count = x_valid.shape[0]
    test_count = x_test.shape[0]

    adjusted_train_count = round(train_percentage / 100 * train_count)
    adjusted_valid_count = round(validation_percentage / 100 * valid_count)
    adjusted_test_count = round(test_percentage / 100 * test_count)

    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = \
        (x_train[:adjusted_train_count], y_train[:adjusted_train_count]), \
        (x_valid[:adjusted_valid_count], y_valid[:adjusted_valid_count]), \
        (x_test[:adjusted_test_count], y_test[:adjusted_test_count])
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

