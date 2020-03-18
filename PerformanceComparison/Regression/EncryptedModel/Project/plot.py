import numpy as np
import matplotlib.pyplot as plt


from PerformanceComparison.Regression.EncryptedModel.Project.consts import *


def plot_loss(losses, title, x_label, y_label, save_directory="", save_title="", show=False):
    plt.plot(losses, "m", label="Training Losses")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if save_directory != "":
        plt.savefig(save_directory + save_title)
    if show:
        plt.show()
    plt.clf()
    return


def plot_regression_outputs(predictions, targets, title, x_label, y_label, save_directory="", save_title="", show=False):
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(predictions, "b", label="Prediction")
    plt.plot(targets, "g", label="Target")
    plt.legend()
    if save_directory != "":
        plt.savefig(save_directory + save_title)
    if show:
        plt.show()
    plt.clf()
    return


def plot_results():
    # Training Output Plotting:
    for epoch_iteration in range(0, TrainingParameters.num_epochs, TrainingParameters.checkpoint_frequency):
        training_output = np.load(training_results_numpy_save_dir + TrainingParameters.training_output_numpy_file_path
                                  + f"{epoch_iteration}.npy")
        y_train = np.load(training_results_numpy_save_dir + TrainingParameters.training_targets_numpy_file_path
                          + f"{epoch_iteration}.npy")
        plot_regression_outputs(training_output, y_train, title=f"Training Output Comparison at Epoch {epoch_iteration}:",
                                x_label="Data Index", y_label="Housing Price in USD",
                                save_directory=training_results_save_dir,
                                save_title=f"training_output_comparison_epoch_{epoch_iteration}.png",
                                show=PlottingParameters.plot_training_outputs)

    # Training Loss Plotting:
    training_losses = np.load(training_results_numpy_save_dir + TrainingParameters.training_losses_numpy_file_path
                              + ".npy")
    plot_loss(training_losses, title=f"Training Losses:", x_label="Epoch Iteration", y_label="Loss",
              save_directory=training_results_save_dir, save_title="training_loss.png",
              show=PlottingParameters.plot_training_loss)


    # Testing Plotting:
    test_output = np.load(testing_results_numpy_save_dir + TestingParameters.testing_numpy_file_path + ".npy")
    y_test = np.load(testing_results_numpy_save_dir + TestingParameters.testing_targets_numpy_file_path + ".npy")
    plot_regression_outputs(test_output, y_test, title="Testing Output Comparison:", x_label="Data Index",
                            y_label="Housing Price in USD", save_directory=testing_results_save_dir,
                            save_title="testing_output.png", show=PlottingParameters.plot_testing_outputs)


if __name__ == "__main__":
    plot_results()
