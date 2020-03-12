from data_loading.load import load_MNIST
from mnist_models.microsoft_cryptonet_model.project.train import train
from mnist_models.microsoft_cryptonet_model.project.test import test
from mnist_models.microsoft_cryptonet_model.project.consts import *

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_MNIST(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        train(x_train, y_train, TrainingParameters.learning_rate, TrainingParameters.num_epochs,
              TrainingParameters.batch_size, checkpoint_frequency=TrainingParameters.checkpoint_frequency,
              num_models=TrainingParameters.num_models, config=EncryptionParameters.config)
    else:
        test(x_test, y_test, TestingParameters.checkpoint_file)
