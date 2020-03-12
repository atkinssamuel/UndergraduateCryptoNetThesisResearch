from encryption.encryption_config import encryption_config

checkpoint_dir = "weights/"
results_dir = "results/"
encrypted_results_dir = "results/encrypted/"
unencrypted_results_dir = "results/unencrypted/"
encrypted_numpy_dir = "results/encrypted_numpy/"
unencrypted_numpy_dir = "results/unencrypted_numpy/"
train_flag = 1
encrypted_flag = 0


class EncryptionParameters:
    if encrypted_flag:
        backend = "HE_SEAL"
        encryption_parameters = ""
    else:
        backend = "CPU"
        encryption_parameters = ""
    config = encryption_config(backend=backend, encryption_parameters=encryption_parameters)


class TrainingParameters:
    learning_rate = 0.001
    num_epochs = 50
    num_models = 100
    batch_size = 2048
    checkpoint_frequency = 2
    training_dataset_percentage = 1


class TestingParameters:
    checkpoint_file = "conv_epoch_48.ckpt"
    testing_dataset_percentage = 100
