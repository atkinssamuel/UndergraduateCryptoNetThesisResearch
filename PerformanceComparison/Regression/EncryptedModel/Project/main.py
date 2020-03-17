from PerformanceComparison.Regression.EncryptedModel.Project.consts import *
from DataManagement.data_loading import load_boston_housing
from PerformanceComparison.Regression.EncryptedModel.Project.train import train
from PerformanceComparison.Regression.EncryptedModel.Project.test import test


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_boston_housing(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        train(x_train, y_train)
    else:
        test(x_test, y_test)
