from Project.consts import *
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
    if Models.EncryptedTiming <= model <= Models.PlaintextTiming:
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
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_MNIST(
            train_percentage=TrainingParameters.training_dataset_percentage,
            validation_percentage=TrainingParameters.valid_dataset_percentage,
            test_percentage=TestingParameters.testing_dataset_percentage)

    if train_flag:
        # Basline Timing Analysis - Boston Housing Dataset
        if model == Models.EncryptedTiming:
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
            simple_classification_train(x_train, y_train, x_valid, y_valid)
        elif model == Models.SomewhatComplexClassification:
            somewhat_complex_classification_train(x_train, y_train)
        elif model == Models.ComplexClassification:
            complex_classification_train(x_train, y_train)
    else:
        # Baseline Timing
        if model == Models.EncryptedTiming:
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
