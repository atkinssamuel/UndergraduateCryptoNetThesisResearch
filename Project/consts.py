from Config.encryption_config import encryption_config
import time
start_time = time.time()


class Models:
    WorkingModelEncrypted = 0
    WorkingModelPlaintext = 1
    LayerWidthInvestigation = 2
    StructureOneLayerSigmoid = 3
    StructureOneLayerSquared = 4
    StructureTwoLayersSigmoid = 5
    StructureTwoLayersSquared = 6
    StructureThreeLayersSigmoid = 7
    StructureThreeLayersSquared = 8
    ScaledSquaredModel = 9
    EncryptedTiming = 10
    PlaintextTiming= 11
    RegressionOneLayer = 12
    RegressionOneLayerPlaintext = 13
    RegressionTwoLayers = 14
    RegressionTwoLayersPlaintext = 15
    RegressionThreeLayers = 16
    RegressionThreeLayersPlaintext = 17
    SimpleClassification = 18
    SomewhatComplexClassification = 19
    ComplexClassification = 20


model = Models.PlaintextTiming
train_flag = 0
encrypted_flag = 0

force_checkpoint = 0
checkpoint_file_number = 650
epochs = 1000
batch_size = 64
learning_rate = 0.001


if model == Models.WorkingModelEncrypted:
    model_dir = "Models/WorkingModelEncrypted/"
    model_name = "WorkingModelEncrypted"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.WorkingModelPlaintext:
    model_dir = "Models/WorkingModelPlaintext/"
    model_name = "WorkingModelPlaintext"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.LayerWidthInvestigation:
    model_dir = "Models/LayerWidthInvestigation/"
    model_name = "LayerWidthInvestigation"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureOneLayerSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSigmoid"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureOneLayerSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSquared"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureTwoLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSigmoid"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureTwoLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSquared"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureThreeLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSigmoid"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureThreeLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSquared"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.ScaledSquaredModel:
    model_dir = "Models/ScaledSquared/"
    model_name = "ScaledSquared"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "ActivationLayerResearch/"
elif model == Models.EncryptedTiming:
    model_dir = "Models/Encrypted/"
    model_name = "EncryptedTiming"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "BaselineTiming/"
elif model == Models.PlaintextTiming:
    model_dir = "Models/Plaintext/"
    model_name = "PlaintextTiming"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Housing Price in 1000's of USD"
    base_dir = "BaselineTiming/"
elif model == Models.RegressionOneLayer:
    model_dir = "Models/RegressionOneLayer/"
    model_name = "RegressionOneLayer"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.RegressionOneLayerPlaintext:
    model_dir = "Models/RegressionOneLayerPlaintext/"
    model_name = "RegressionOneLayerPlaintext"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.RegressionTwoLayers:
    model_dir = "Models/RegressionTwoLayers/"
    model_name = "RegressionTwoLayers"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.RegressionTwoLayersPlaintext:
    model_dir = "Models/RegressionTwoLayersPlaintext/"
    model_name = "RegressionTwoLayersPlaintext"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.RegressionThreeLayers:
    model_dir = "Models/RegressionThreeLayers/"
    model_name = "RegressionThreeLayers"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.RegressionThreeLayersPlaintext:
    model_dir = "Models/RegressionThreeLayersPlaintext/"
    model_name = "RegressionThreeLayersPlaintext"
    model_type = "Regression"
    plot_x_label = "Data Index"
    plot_y_label = "Year Song Released"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.SimpleClassification:
    model_dir = "Models/SimpleClassification/"
    model_name = "SimpleClassification"
    model_type = "Classification"
    plot_x_label = "Data Index"
    plot_y_label = ""
    base_dir = "PerformanceComparison/Classification/"
elif model == Models.SomewhatComplexClassification:
    model_dir = "Models/SomewhatComplexClassification/"
    model_name = "SomewhatComplexClassification"
    model_type = "Classification"
    plot_x_label = "Data Index"
    plot_y_label = ""
    base_dir = "PerformanceComparison/Classification/"
elif model == Models.ComplexClassification:
    model_dir = "Models/ComplexClassification/"
    model_name = "ComplexClassification"
    model_type = "Classification"
    plot_x_label = "Data Index"
    plot_y_label = ""
    base_dir = "PerformanceComparison/Classification/"
else:
    print("Invalid model enum.")
    exit(1)

checkpoint_dir = base_dir + model_dir + "Weights/"
results_dir = base_dir + model_dir + "Results/"
training_results_save_dir = base_dir + model_dir + "Results/Training/"
training_results_numpy_save_dir = base_dir + model_dir + "Results/TrainingNumpy/"
testing_results_numpy_save_dir = base_dir + model_dir + "Results/TestingNumpy/"
testing_results_save_dir = base_dir + model_dir + "Results/Testing/"
config_base_options_dir = "Config/BaseOptions/"
config_options_dir = "Config/Options/"



class BackendOptions:
    SEAL = "HE_SEAL"
    CPU = "CPU"
    XLA = "XLA"


class EncryptionParameters:
    if encrypted_flag:
        print("Model is encrypted. Using corresponding encryption parameters.")
        backend = BackendOptions.SEAL
        encryption_parameters = config_options_dir + "base_params.json"
    else:
        print("Model is unencrypted.")
        backend = BackendOptions.CPU
        encryption_parameters = ""
    print("encryption_parameters =", encryption_parameters)
    config = encryption_config(backend=backend, encryption_parameters=encryption_parameters)

class TrainingParameters:
    learning_rate = learning_rate
    num_epochs = epochs
    num_models = 100
    batch_size = batch_size
    checkpoint_frequency = 5
    incomplete_checkpoint_file_location = checkpoint_dir + model_type + "_" + model_name + "_Epoch_"
    training_dataset_percentage = 100
    valid_dataset_percentage = 100
    training_output_numpy_file_path = "training_output_epoch_"
    training_targets_numpy_file_path = "training_targets_epoch_"
    training_losses_numpy_file_path = "training_losses"
    validation_losses_numpy_file_path = "validation_losses"


class TestingParameters:
    checkpoint_file_number = f"{checkpoint_file_number}"
    checkpoint_file_location = TrainingParameters.incomplete_checkpoint_file_location + \
                               checkpoint_file_number + ".ckpt"
    testing_dataset_percentage = 100
    testing_output_numpy_file_path = "testing_output"
    testing_targets_numpy_file_path = "testing_targets"


class PlottingParameters:
    plot_training_outputs = False
    plot_training_loss = True
    plot_testing_outputs = True
    testing_output_plot_percentage = 20
