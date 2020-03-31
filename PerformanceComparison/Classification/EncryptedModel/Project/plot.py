import matplotlib.pyplot as plt
import numpy as np
from PerformanceComparison.Classification.EncryptedModel.Project.consts import *

if __name__ == "__main__":
    training_losses = np.load(training_results_numpy_dir + "training_losses.npy")
    training_accuracies = np.load(training_results_numpy_dir + "training_accuracies.npy")

    # Loss Plotting:
    plt.title("Training Loss:")
    plt.ylabel("Loss")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_losses)
    plt.savefig(training_results_dir + "training_loss.png")
    plt.show()

    # Accuracy Plotting:
    plt.title("Training Accuracy:")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_accuracies, "g")
    plt.savefig(training_results_dir + "training_accuracy.png")
    plt.show()
