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


model = Models.ScaledSquaredModel
train_flag = 0
encrypted_flag = not train_flag
checkpoint_file_number = 1480

base_dir = "BostonHousingRegressionNets/"

if model == Models.WorkingModelEncrypted:
    model_dir = "Models/WorkingModelEncrypted/"
    model_name = "WorkingModelEncrypted"
elif model == Models.WorkingModelPlaintext:
    model_dir = "Models/WorkingModelPlaintext/"
    model_name = "WorkingModelPlaintext"
elif model == Models.LayerWidthInvestigation:
    model_dir = "Models/LayerWidthInvestigation/"
    model_name = "LayerWidthInvestigation"
elif model == Models.StructureOneLayerSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSigmoid"
elif model == Models.StructureOneLayerSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureOneLayerSquared"
elif model == Models.StructureTwoLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSigmoid"
elif model == Models.StructureTwoLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureTwoLayersSquared"
elif model == Models.StructureThreeLayersSigmoid:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSigmoid"
elif model == Models.StructureThreeLayersSquared:
    model_dir = "Models/StructureInvestigation/"
    model_name = "StructureThreeLayersSquared"
elif model == Models.ScaledSquaredModel:
    model_dir = "Models/ScaledSquared/"
    model_name = "ScaledSquared"

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
model_type = "BostonHousingRegression"



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
