from mnist import MNIST
from keras.datasets import cifar10, cifar100, fashion_mnist, boston_housing
import numpy as np
from DataManagement.data_constants import *


def import_MNIST():
    # Ensure you replace the dots in the dataset files with "-"s to avoid a "File not found" error
    MNIST_data = MNIST(MNIST_constants.mnist_dataset_dir)

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

    np.save(MNIST_constants.mnist_np_save_dir + 'x_train', x_train)
    np.save(MNIST_constants.mnist_np_save_dir + 'y_train', y_train_categorical)
    np.save(MNIST_constants.mnist_np_save_dir + 'x_test', x_test)
    np.save(MNIST_constants.mnist_np_save_dir + 'y_test', y_test_categorical)
    return


def import_cifar100():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    np.save(cifar100_constants.cifar100_np_save_dir + 'x_train', x_train)
    np.save(cifar100_constants.cifar100_np_save_dir + 'y_train', y_train)
    np.save(cifar100_constants.cifar100_np_save_dir + 'x_test', x_test)
    np.save(cifar100_constants.cifar100_np_save_dir + 'y_test', y_test)
    return


def import_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    np.save(cifar10_constants.cifar10_np_save_dir + 'x_train', x_train)
    np.save(cifar10_constants.cifar10_np_save_dir + 'y_train', y_train)
    np.save(cifar10_constants.cifar10_np_save_dir + 'x_test', x_test)
    np.save(cifar10_constants.cifar10_np_save_dir + 'y_test', y_test)
    return

def import_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    np.save(fashion_mnist_constants.fashion_mnist_np_save_dir + 'x_train', x_train)
    np.save(fashion_mnist_constants.fashion_mnist_np_save_dir + 'y_train', y_train)
    np.save(fashion_mnist_constants.fashion_mnist_np_save_dir + 'x_test', x_test)
    np.save(fashion_mnist_constants.fashion_mnist_np_save_dir + 'y_test', y_test)
    return


def import_boston_housing():
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    np.save(boston_housing_constants.boston_housing_np_save_dir + 'x_train', x_train)
    np.save(boston_housing_constants.boston_housing_np_save_dir + 'y_train', y_train)
    np.save(boston_housing_constants.boston_housing_np_save_dir + 'x_test', x_test)
    np.save(boston_housing_constants.boston_housing_np_save_dir + 'y_test', y_test)
    return


if __name__ == "__main__":
    import_boston_housing()

