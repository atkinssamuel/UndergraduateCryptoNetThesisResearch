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
# Baseline Timing:
from BaselineTiming.Models.Encrypted.encrypted_train import encrypted_train
from BaselineTiming.Models.Encrypted.encrypted_test import encrypted_test
from BaselineTiming.Models.Plaintext.plaintext_train import plaintext_train
from BaselineTiming.Models.Plaintext.plaintext_test import plaintext_test
# Performance Comparison Imports:
# One Layer:
from PerformanceComparison.Regression.Models.RegressionOneLayer.regression_one_layer_train \
    import regression_one_layer_train
from PerformanceComparison.Regression.Models.RegressionOneLayer.regression_one_layer_test \
    import regression_one_layer_test
from PerformanceComparison.Regression.Models.RegressionOneLayerPlaintext.regression_one_layer_plaintext_train \
    import regression_one_layer_plaintext_train
from PerformanceComparison.Regression.Models.RegressionOneLayerPlaintext.regression_one_layer_plaintext_test \
    import regression_one_layer_plaintext_test
# Two Layers:
from PerformanceComparison.Regression.Models.RegressionTwoLayers.regression_two_layers_train \
    import regression_two_layers_train
from PerformanceComparison.Regression.Models.RegressionTwoLayers.regression_two_layers_test \
    import regression_two_layers_test
from PerformanceComparison.Regression.Models.RegressionTwoLayersPlaintext.regression_two_layers_plaintext_train \
    import regression_two_layers_plaintext_train
from PerformanceComparison.Regression.Models.RegressionTwoLayersPlaintext.regression_two_layers_plaintext_test \
    import regression_two_layers_plaintext_test
# Three Layers:
from PerformanceComparison.Regression.Models.RegressionThreeLayers.regression_three_layers_train \
    import regression_three_layers_train
from PerformanceComparison.Regression.Models.RegressionThreeLayers.regression_three_layers_test \
    import regression_three_layers_test
from PerformanceComparison.Regression.Models.RegressionThreeLayersPlaintext.regression_three_layers_plaintext_train \
    import regression_three_layers_plaintext_train
from PerformanceComparison.Regression.Models.RegressionThreeLayersPlaintext.regression_three_layers_plaintext_test \
    import regression_three_layers_plaintext_test

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
    if Models.WorkingModelEncrypted <= model <= Models.ScaledSquaredModel:
        (x_train, y_train), (x_test, y_test) = load_boston_housing(
            train_percentage=TrainingParameters.training_dataset_percentage,
            test_percentage=TestingParameters.testing_dataset_percentage)
    elif Models.EncryptedTiming <= model <= Models.PlaintextTiming:
        (x_train, y_train), (x_test, y_test) = load_boston_housing(
            train_percentage=TrainingParameters.training_dataset_percentage,
            test_percentage=TestingParameters.testing_dataset_percentage)
        # Using the entire dataset for testing:
        x_test = np.vstack((x_train, x_test))
        y_test = np.vstack((y_train, y_test))

    elif Models.RegressionOneLayer <= model <= Models.RegressionThreeLayers:
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_year_prediction(
            train_percentage=TrainingParameters.training_dataset_percentage,
            validation_percentage=TrainingParameters.valid_dataset_percentage,
            test_percentage=TestingParameters.testing_dataset_percentage)
    elif Models.SimpleClassification <= model <= Models.ComplexClassification:
        (x_train, y_train), (x_test, y_test) = load_MNIST(
            train_percentage=TrainingParameters.training_dataset_percentage,
            test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        # Simple Working Models - Boston Housing Dataset
        if model == Models.WorkingModelEncrypted:
            working_model_encrypted_train(x_train, y_train)
        elif model == Models.WorkingModelPlaintext:
            working_model_plaintext_train(x_train, y_train)
        # Layer Width Investigation - Boston Housing Dataset
        elif model == Models.LayerWidthInvestigation:
            layer_width_investigation_train(x_train, y_train)
        # Structure Investigation - Boston Housing Dataset
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
        # Scaled Squared Model - Boston Housing Dataset
        elif model == Models.ScaledSquaredModel:
            scaled_squared_train(x_train, y_train)
        # Basline Timing Analysis - Boston Housing Dataset
        elif model == Models.EncryptedTiming:
            encrypted_train(x_train, y_train)
        elif model == Models.PlaintextTiming:
            plaintext_train(x_train, y_train)
        # Regression Comparison - Year Prediction Dataset
        elif model == Models.RegressionOneLayer:
            regression_one_layer_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.RegressionOneLayerPlaintext:
            regression_one_layer_plaintext_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.RegressionTwoLayers:
            regression_two_layers_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.RegressionTwoLayersPlaintext:
            regression_two_layers_plaintext_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.RegressionThreeLayers:
            regression_three_layers_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.RegressionThreeLayersPlaintext:
            regression_three_layers_plaintext_train(x_train, y_train, x_valid, y_valid)
        # Classification Comparison - MNIST Dataset
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
        # Baseline Timing
        elif model == Models.EncryptedTiming:
            encrypted_test(x_test, y_test)
        elif model == Models.PlaintextTiming:
            plaintext_test(x_test, y_test)
        # Regression Comparison
        elif model == Models.RegressionOneLayer:
            regression_one_layer_test(x_test, y_test)
        elif model == Models.RegressionOneLayerPlaintext:
            regression_one_layer_plaintext_test(x_test, y_test)
        elif model == Models.RegressionTwoLayers:
            regression_two_layers_test(x_test, y_test)
        elif model == Models.RegressionTwoLayersPlaintext:
            regression_two_layers_plaintext_test(x_test, y_test)
        elif model == Models.RegressionThreeLayers:
            regression_three_layers_test(x_test, y_test)
        elif model == Models.RegressionThreeLayersPlaintext:
            regression_three_layers_plaintext_test(x_test, y_test)
        # Classification Comparison
        elif model == Models.SimpleClassification:
            simple_classification_test(x_test, y_test)
        elif model == Models.SomewhatComplexClassification:
            somewhat_complex_classification_test(x_test, y_test)
        elif model == Models.ComplexClassification:
            complex_classification_test(x_test, y_test)
        elif model == Models.ScaledSquaredModel:
            scaled_squared_test(x_test, y_test)