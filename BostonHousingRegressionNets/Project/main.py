from BostonHousingRegressionNets.Project.consts import  *
from BostonHousingRegressionNets.Models.WorkingModel.working_model_train \
    import working_model_train
from BostonHousingRegressionNets.Models.ThresholdModel.threshold_model_train \
    import threshold_model_train
from BostonHousingRegressionNets.Models.ExplodedModel.exploded_model_train \
    import exploded_model_train
from BostonHousingRegressionNets.Models.WorkingModel.working_model_test import \
    working_model_test
from BostonHousingRegressionNets.Models.ThresholdModel.threshold_model_test import \
    threshold_model_test
from BostonHousingRegressionNets.Models.ExplodedModel.exploded_model_test import \
    exploded_model_test


from DataManagement.data_loading import load_boston_housing


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_boston_housing(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        if model == Models.WorkingModel:
            working_model_train(x_train, y_train)
        elif model == Models.ThresholdModel:
            threshold_model_train(x_train, y_train)
        elif model == Models.ExplodedModel:
            exploded_model_train(x_train, y_train)
    else:
        if model == Models.WorkingModel:
            working_model_test(x_test, y_test)
        elif model == Models.ThresholdModel:
            threshold_model_test(x_train, y_train)
        elif model == Models.ExplodedModel:
            exploded_model_test(x_train, y_train)
