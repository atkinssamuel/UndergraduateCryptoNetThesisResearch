from BostonHousingRegressionNets.ExplodingGradientIllustration.Project.consts import  *
from BostonHousingRegressionNets.ExplodingGradientIllustration.Models.train import train
from BostonHousingRegressionNets.ExplodingGradientIllustration.Models.test import test
from DataManagement.data_loading import load_boston_housing


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_boston_housing(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        train(x_train, y_train)
    else:
        test(x_test, y_test)
