from DataManagement.data_loading import load_MNIST
from PerformanceComparison.Classification.EncryptedModel.Project.train import train
from PerformanceComparison.Classification.EncryptedModel.Project.test import test
from PerformanceComparison.Classification.EncryptedModel.Project.consts import *

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
