from Config.encryption_config import encryption_config
import time
start_time = time.time()


class Models:
    EncryptedTiming = 1
    PlaintextTiming= 2
    RegressionOneLayer = 3
    RegressionOneLayerPlaintext = 4
    RegressionTwoLayers = 5
    RegressionTwoLayersPlaintext = 6
    RegressionThreeLayers = 7
    RegressionThreeLayersPlaintext = 8
    SimpleClassification = 9
    SomewhatComplexClassification = 10
    ComplexClassification = 11


model = Models.RegressionTwoLayers
train_flag = 1
encrypted_flag = 1

force_checkpoint = 0
checkpoint_file_number = 105
epochs = 1000
batch_size = 64
learning_rate = 0.005


if model == Models.EncryptedTiming:
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
