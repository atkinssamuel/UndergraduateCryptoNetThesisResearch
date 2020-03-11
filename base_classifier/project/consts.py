base_classifier_checkpoint_dir = "base_classifier/weights"
base_classifier_results_dir = "base_classifier/results"


class TrainingParameters:
    learning_rate = 0.005
    num_epochs = 50
    num_models = 100
    batch_size = 2048
    checkpoint_frequency = 2
