from data_management.data_manipulation import import_MNIST, load_MNIST, load_MNIST_flat
from base_classifier.project.train import train
from base_classifier.project.test import test
from base_classifier.project.consts import base_classifier_results_dir, base_classifier_checkpoint_dir
from base_classifier.project.consts import TrainingParameters


if __name__ == "__main__":
    _train = 1
    checkpoint_file = "conv_epoch_48.ckpt"
    (x_train, y_train), (x_test, y_test) = load_MNIST_flat(train_percentage=10, test_percentage=100)

    if _train:
        train(x_train, y_train, TrainingParameters.learning_rate, TrainingParameters.num_epochs,
              TrainingParameters.batch_size, checkpoint_frequency=TrainingParameters.checkpoint_frequency,
              num_models=TrainingParameters.num_models)
    else:
        test(x_test, y_test, checkpoint_file)
