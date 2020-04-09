from Project.consts import *
from BostonHousingRegressionNets.Models.WorkingModelEncrypted.working_model_train \
    import working_model_encrypted_train
from BostonHousingRegressionNets.Models.WorkingModelPlaintext.working_model_plaintext_train \
    import working_model_plaintext_train
from BostonHousingRegressionNets.Models.LayerWidthInvestigation.layer_width_investigation_train \
    import layer_width_investigation_train
from BostonHousingRegressionNets.Models.WorkingModelEncrypted.working_model_test import \
    working_model_encrypted_test
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
from BostonHousingRegressionNets.Models.ScaledSquared.scaled_squared_train import scaled_squared_train
from BostonHousingRegressionNets.Models.ScaledSquared.scaled_squared_test import scaled_squared_test


from DataManagement.data_loading import load_boston_housing

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_boston_housing(
        train_percentage=TrainingParameters.training_dataset_percentage,
        test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        # Simple Working Models
        if model == Models.WorkingModelEncrypted:
            working_model_encrypted_train(x_train, y_train)
        elif model == Models.WorkingModelPlaintext:
            working_model_plaintext_train(x_train, y_train)
        # Layer Width Investigation
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
        # Scaled Squared Model
        elif model == Models.ScaledSquaredModel:
            scaled_squared_train(x_train, y_train)

    else:
        if model == Models.WorkingModelEncrypted:
            working_model_encrypted_test(x_test, y_test)
        elif model == Models.WorkingModelPlaintext:
            working_model_plaintext_test(x_test, y_test)
        elif model == Models.LayerWidthInvestigation:
            print("No test file for LayerWidthInvestigation.")
            exit(1)
        elif Models.StructureOneLayerSigmoid <= model <= Models.StructureThreeLayersSquared:
            print("No test files for Structure Investigation.")
            exit(1)
        elif model == Models.ScaledSquaredModel:
            scaled_squared_test(x_test, y_test)