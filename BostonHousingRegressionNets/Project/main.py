from BostonHousingRegressionNets.Project.consts import *
from BostonHousingRegressionNets.Models.WorkingModel.working_model_train \
    import working_model_train
from BostonHousingRegressionNets.Models.WorkingModelPlaintext.working_model_plaintext_train \
    import working_model_plaintext_train
from BostonHousingRegressionNets.Models.LayerWidthInvestigation.layer_width_investigation_train \
    import layer_width_investigation_train
from BostonHousingRegressionNets.Models.WorkingModel.working_model_test import \
    working_model_test
from BostonHousingRegressionNets.Models.WorkingModelPlaintext.working_model_plaintext_test \
    import working_model_plaintext_test
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.one_layer_train_sigmoid import \
    one_layer_train_sigmoid
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.one_layer_train_squared import \
    one_layer_train_squared
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.two_layers_train_sigmoid import \
    two_layers_train_sigmoid
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.two_layers_train_squared import \
    two_layers_train_squared
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.three_layers_train_sigmoid import \
    three_layers_train_sigmoid
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.three_layers_train_squared import \
    three_layers_train_squared
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.four_layers_train_sigmoid import \
    four_layers_train_sigmoid
from BostonHousingRegressionNets.Models.StructureInvestigation.Structures.four_layers_train_squared import \
    four_layers_train_squared

from DataManagement.data_loading import load_boston_housing

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_boston_housing(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        if model == Models.WorkingModel:
            working_model_train(x_train, y_train)
        elif model == Models.WorkingModelPlaintext:
            working_model_plaintext_train(x_train, y_train)
        elif model == Models.LayerWidthInvestigation:
            layer_width_investigation_train(x_train, y_train)
        # Structure Investigation
        elif model == Models.StructureOneLayerSigmoid:
            one_layer_train_sigmoid(x_train, y_train)
        elif model == Models.StructureOneLayerSquared:
            one_layer_train_squared(x_train, y_train)
        elif model == Models.StructureTwoLayersSigmoid:
            two_layers_train_sigmoid(x_train, y_train)
        elif model == Models.StructureTwoLayersSquared:
            two_layers_train_squared(x_train, y_train)
        elif model == Models.StructureThreeLayersSigmoid:
            three_layers_train_sigmoid(x_train, y_train)
        elif model == Models.StructureThreeLayersSquared:
            three_layers_train_squared(x_train, y_train)
        elif model == Models.StructureFourLayersSquared:
            four_layers_train_sigmoid(x_train, y_train)
        elif model == Models.StructureFourLayersSquared:
            four_layers_train_squared(x_train, y_train)
    else:
        if model == Models.WorkingModel:
            working_model_test(x_test, y_test)
        elif model == Models.WorkingModelPlaintext:
            working_model_plaintext_test(x_test, y_test)
        elif model == Models.LayerWidthInvestigation:
            print("No test file for LayerWidthInvestigation.")
            exit(1)
        elif Models.StructureOneLayerSigmoid <= model <= Models.StructureFourLayersSquared:
            print("No test files for Structure Investigation.")
            exit(1)
