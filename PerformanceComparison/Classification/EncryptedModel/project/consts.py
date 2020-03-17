from encryption.encryption_config import encryption_config

checkpoint_dir = "weights/"
results_dir = "results/"
encrypted_results_dir = "results/encrypted/"
unencrypted_results_dir = "results/unencrypted/"
encrypted_numpy_dir = "results/encrypted_numpy/"
unencrypted_numpy_dir = "results/unencrypted_numpy/"
train_flag = 1
encrypted_flag = 0


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
    num_epochs = 300
    num_models = 100
    batch_size = 64
    checkpoint_frequency = 2
    training_dataset_percentage = 100


class TestingParameters:
    checkpoint_file = "conv_epoch_48.ckpt"
    testing_dataset_percentage = 100
