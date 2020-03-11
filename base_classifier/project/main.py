from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from base_classifier.project.train import train
from base_classifier.project.test import test

def run_network(_train, x_train, y_train, x_test, y_test):
    # Training Parameters:
    learning_rate = 0.0001
    num_epochs = 50
    num_models = 100
    batch_size = 512
    checkpoint_frequency = 2
    # Testing Parameters:
    checkpoint_file = "conv_epoch_48.ckpt"
    if _train:
        train(x_train, y_train, learning_rate, num_epochs, batch_size, checkpoint_frequency=checkpoint_frequency,
                    num_models=num_models)
    else:
        test(x_test, y_test, checkpoint_file)


if __name__ == "__main__":
    _train = 1
    (x_train, y_train), (x_test, y_test) = load_MNIST(train_percentage=10, test_percentage=100)
    run_network(_train, x_train, y_train, x_test, y_test)

