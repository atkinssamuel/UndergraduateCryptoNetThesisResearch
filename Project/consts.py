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
    SimpleRegression = 10
    SomewhatComplexRegression = 11
    ComplexRegression = 12
    SimpleClassification = 13
    SomewhatComplexClassification = 14
    ComplexClassification = 15


model = Models.SimpleRegression
train_flag = 1
encrypted_flag = not train_flag
checkpoint_file_number = 1480


if model == Models.WorkingModelEncrypted:
    model_dir = "Models/WorkingModelEncrypted/"
    model_name = "WorkingModelEncrypted"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.WorkingModelPlaintext:
    model_dir = "Models/WorkingModelPlaintext/"
    model_name = "WorkingModelPlaintext"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.LayerWidthInvestigation:
    model_dir = "Models/LayerWidthInvestigation/"
    model_name = "LayerWidthInvestigation"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureOneLayerSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSigmoid"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureOneLayerSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSquared"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureTwoLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSigmoid"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureTwoLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSquared"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureThreeLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSigmoid"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.StructureThreeLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSquared"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.ScaledSquaredModel:
    model_dir = "Models/ScaledSquared/"
    model_name = "ScaledSquared"
    model_type = "Regression"
    base_dir = "ActivationLayerResearch/"
elif model == Models.SimpleRegression:
    model_dir = "Models/SimpleRegression/"
    model_name = "SimpleRegression"
    model_type = "Regression"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.SomewhatComplexRegression:
    model_dir = "Models/SomewhatComplexRegression/"
    model_name = "SomewhatComplexRegression"
    model_type = "Regression"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.ComplexRegression:
    model_dir = "Models/ComplexRegression/"
    model_name = "ComplexRegression"
    model_type = "Regression"
    base_dir = "PerformanceComparison/Regression/"
elif model == Models.SimpleClassification:
    model_dir = "Models/SimpleClassification/"
    model_name = "SimpleClassification"
    model_type = "Classification"
    base_dir = "PerformanceComparison/Classification/"
elif model == Models.SomewhatComplexClassification:
    model_dir = "Models/SomewhatComplexClassification/"
    model_name = "SomewhatComplexClassification"
    model_type = "Classification"
    base_dir = "PerformanceComparison/Classification/"
elif model == Models.ComplexClassification:
    model_dir = "Models/ComplexClassification/"
    model_name = "ComplexClassification"
    model_type = "Classification"
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
    learning_rate = 0.001
    num_epochs = 1500
    num_models = 100
    batch_size = 64
    checkpoint_frequency = 10
    incomplete_checkpoint_file_location = checkpoint_dir + model_type + "_" + model_name + "_Epoch_"
    training_dataset_percentage = 100
    training_output_numpy_file_path = "training_output_epoch_"
    training_targets_numpy_file_path = "training_targets_epoch_"
    training_losses_numpy_file_path = "training_losses"


class TestingParameters:
    checkpoint_file_number = f"{checkpoint_file_number}"
    checkpoint_file_location = TrainingParameters.incomplete_checkpoint_file_location + \
                               checkpoint_file_number + ".ckpt"
    testing_dataset_percentage = 100
    testing_numpy_file_path = "testing_output"
    testing_targets_numpy_file_path = "testing_targets"


class PlottingParameters:
    plot_training_outputs = False
    plot_training_loss = True
    plot_testing_outputs = True
