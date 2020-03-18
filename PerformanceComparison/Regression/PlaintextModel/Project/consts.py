base_dir = "PerformanceComparison/Regression/PlaintextModel/"
checkpoint_dir = base_dir + "Weights/"
results_dir = base_dir + "Results/"
training_results_save_dir = base_dir + "Results/Training/"
testing_results_save_dir = base_dir + "Results/Testing/"
model_type = "Regression"
model_name = "PlaintextModel"
train_flag = 1


class TrainingParameters:
    learning_rate = 0.001
    num_epochs = 1500
    num_models = 100
    batch_size = 128
    checkpoint_frequency = 100
    incomplete_checkpoint_file_location = checkpoint_dir + model_type + "_" + model_name + "_Epoch_"
    training_dataset_percentage = 100


class TestingParameters:
    checkpoint_file_number = "1400"
    checkpoint_file_location = TrainingParameters.incomplete_checkpoint_file_location + \
                               checkpoint_file_number + ".ckpt"
    testing_dataset_percentage = 100
