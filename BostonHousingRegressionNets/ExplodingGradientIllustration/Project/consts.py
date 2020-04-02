from GlobalHelpers.encryption_config import encryption_config


class Models:
    WorkingModel = 0
    ThresholdModel = 1
    ExplodedModel = 2


model = Models.WorkingModel
train_flag = 0
encrypted_flag = 0

base_dir = "BostonHousingRegressionNets/ExplodingGradientIllustration/"

if model == Models.WorkingModel:
    model_dir = "Models/WorkingModel/"
elif model == Models.ThresholdModel:
    model_dir = "Models/ThresholdModel/"
elif model == Models.ExplodedModel:
    model_dir = "Models/ExplodedModel/"
else:
    print("Invalid model enum.")
    exit(1)

checkpoint_dir = base_dir + model_dir + "Weights/"
results_dir = base_dir + model_dir + "Results/"
training_results_save_dir = base_dir + model_dir + "Results/Training/"
training_results_numpy_save_dir = base_dir + model_dir + "Results/TrainingNumpy/"
testing_results_numpy_save_dir = base_dir + model_dir + "Results/TestingNumpy/"
testing_results_save_dir = base_dir + model_dir + "Results/Testing/"
model_type = "BostonHousingRegression"
model_name = "ExplodingGradient"


class BackendOptions:
    SEAL = "HE_SEAL"
    CPU = "CPU"
    XLA = "XLA"


class EncryptionParameters:
    if encrypted_flag:
        backend = BackendOptions.SEAL
        encryption_parameters = ""
    else:
        backend = BackendOptions.XLA
        encryption_parameters = ""
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
    checkpoint_file_number = "1430"
    checkpoint_file_location = TrainingParameters.incomplete_checkpoint_file_location + \
                               checkpoint_file_number + ".ckpt"
    testing_dataset_percentage = 100
    testing_numpy_file_path = "testing_output"
    testing_targets_numpy_file_path = "testing_targets"


class PlottingParameters:
    plot_training_outputs = False
    plot_training_loss = True
    plot_testing_outputs = True
