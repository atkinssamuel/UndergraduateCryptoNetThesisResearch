from Project.consts import *
# ActivationLayerResearchImports:
from ActivationLayerResearch.Models.WorkingModelEncrypted.working_model_train \
    import working_model_encrypted_train
from ActivationLayerResearch.Models.WorkingModelPlaintext.working_model_plaintext_train \
    import working_model_plaintext_train
from ActivationLayerResearch.Models.LayerWidthInvestigation.layer_width_investigation_train \
    import layer_width_investigation_train
from ActivationLayerResearch.Models.WorkingModelEncrypted.working_model_test import \
    working_model_encrypted_test
from ActivationLayerResearch.Models.WorkingModelPlaintext.working_model_plaintext_test \
    import working_model_plaintext_test
from ActivationLayerResearch.Models.StructureInvestigation.Structures.one_layer_train_sigmoid import \
    one_layer_train_sigmoid
from ActivationLayerResearch.Models.StructureInvestigation.Structures.one_layer_train_squared import \
    one_layer_train_squared
from ActivationLayerResearch.Models.StructureInvestigation.Structures.two_layers_train_sigmoid import \
    two_layers_train_sigmoid
from ActivationLayerResearch.Models.StructureInvestigation.Structures.two_layers_train_squared import \
    two_layers_train_squared
from ActivationLayerResearch.Models.StructureInvestigation.Structures.three_layers_train_sigmoid import \
    three_layers_train_sigmoid
from ActivationLayerResearch.Models.StructureInvestigation.Structures.three_layers_train_squared import \
    three_layers_train_squared
from ActivationLayerResearch.Models.ScaledSquared.scaled_squared_train import scaled_squared_train
from ActivationLayerResearch.Models.ScaledSquared.scaled_squared_test import scaled_squared_test
# Performance Comparison Imports:
from PerformanceComparison.Regression.Models.ComplexRegression.complex_regression_train import complex_regression_train
from PerformanceComparison.Regression.Models.ComplexRegression.complex_regression_test import complex_regression_test
from PerformanceComparison.Regression.Models.SomewhatComplexRegression.somewhat_complex_regression_train \
    import somewhat_complex_regression_train
from PerformanceComparison.Regression.Models.SomewhatComplexRegression.somewhat_complex_regression_test \
    import somewhat_complex_regression_test
from PerformanceComparison.Regression.Models.SimpleRegression.simple_regression_train import simple_regression_train
from PerformanceComparison.Regression.Models.SimpleRegression.simple_regression_test import simple_regression_test
from PerformanceComparison.Classification.Models.ComplexClassification.complex_classification_train \
    import complex_classification_train
from PerformanceComparison.Classification.Models.ComplexClassification.complex_classification_test import complex_classification_test
from PerformanceComparison.Classification.Models.SomewhatComplexClassification.somewhat_complex_classification_train \
    import somewhat_complex_classification_train
from PerformanceComparison.Classification.Models.SomewhatComplexClassification.somewhat_complex_classification_test \
    import somewhat_complex_classification_test
from PerformanceComparison.Classification.Models.SimpleClassification.simple_classification_train import simple_classification_train
from PerformanceComparison.Classification.Models.SimpleClassification.simple_classification_test import simple_classification_test
from DataManagement.data_loading import *


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_year_prediction(
        train_percentage=TrainingParameters.training_dataset_percentage,
        validation_percentage=TrainingParameters.valid_dataset_percentage,
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
        # Regression Comparison
        elif model == Models.SimpleRegression:
            simple_regression_train(x_train, y_train)
        elif model == Models.SomewhatComplexRegression:
            somewhat_complex_regression_train(x_train, y_train)
        elif model == Models.ComplexRegression:
            complex_regression_train(x_train, y_train)
        # Classification Comparison
        elif model == Models.SimpleClassification:
            simple_classification_train(x_train, y_train)
        elif model == Models.SomewhatComplexClassification:
            somewhat_complex_classification_train(x_train, y_train)
        elif model == Models.ComplexClassification:
            complex_classification_train(x_train, y_train)
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
        # Regression Comparison
        elif model == Models.SimpleRegression:
            simple_regression_test(x_train, y_train)
        elif model == Models.SomewhatComplexRegression:
            somewhat_complex_regression_test(x_train, y_train)
        elif model == Models.ComplexRegression:
            complex_regression_test(x_train, y_train)
        # Classification Comparison
        elif model == Models.SimpleClassification:
            simple_classification_test(x_train, y_train)
        elif model == Models.SomewhatComplexClassification:
            somewhat_complex_classification_test(x_train, y_train)
        elif model == Models.ComplexClassification:
            complex_classification_test(x_train, y_train)
        elif model == Models.ScaledSquaredModel:
            scaled_squared_test(x_test, y_test)