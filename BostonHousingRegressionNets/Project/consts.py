from Config.encryption_config import encryption_config
import time
start_time = time.time()


class Models:
    WorkingModel = 0
    WorkingModelPlaintext = 1
    ThresholdModel = 2
    ExplodedModel = 3


model = Models.ExplodedModel
train_flag = 1
encrypted_flag = not train_flag
checkpoint_file_number = 510

base_dir = "BostonHousingRegressionNets/"

if model == Models.WorkingModel:
    model_dir = "Models/WorkingModel/"
    model_name = "WorkingModel"
elif model == Models.WorkingModelPlaintext:
    model_dir = "Models/WorkingModelPlaintext/"
    model_name = "WorkingModelPlaintext"
elif model == Models.ThresholdModel:
    model_dir = "Models/ThresholdModel/"
    model_name = "ThresholdModel"
elif model == Models.ExplodedModel:
    model_dir = "Models/ExplodedModel/"
    model_name = "ExplodedModel"
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
    num_epochs = 2
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
