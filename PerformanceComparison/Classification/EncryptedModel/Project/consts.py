from GlobalHelpers.encryption_config import encryption_config

base_dir = "PerformanceComparison/Classification/EncryptedModel/"
checkpoint_dir = base_dir + "Weights/"
results_dir = base_dir + "Results/"
training_results_dir = base_dir + "Results/Training/"
training_results_numpy_dir = base_dir + "Results/TrainingNumpy/"
testing_results_dir = base_dir + "Results/Testing/"
testing_results_numpy_dir = base_dir + "Results/TestingNumpy/"
model_type = "Classification"
model_name = "EncryptedModel"

train_flag = 0
encrypted_flag = 1


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
    learning_rate = 0.01
    num_epochs = 10
    num_models = 100
    batch_size = 64
    checkpoint_frequency = 2
    training_dataset_percentage = 100
    incomplete_checkpoint_file_location = checkpoint_dir + model_type + "_" + model_name + "_Epoch_"



class TestingParameters:
    checkpoint_file = "conv_epoch_48.ckpt"
    testing_dataset_percentage = 100
    checkpoint_file_number = "6"
    checkpoint_file_location = TrainingParameters.incomplete_checkpoint_file_location + \
                               checkpoint_file_number + ".ckpt"
