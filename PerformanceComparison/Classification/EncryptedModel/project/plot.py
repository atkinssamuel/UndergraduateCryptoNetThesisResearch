import matplotlib.pyplot as plt
import numpy as np
from mnist_models.microsoft_cryptonet_model.project.consts import *

if __name__ == "__main__":
    if encrypted_flag:
        training_losses = np.load(encrypted_numpy_dir + "encrypted_training_losses.npy")
        training_accuracies = np.load(encrypted_numpy_dir + "encrypted_training_accuracies.npy")
        save_dir = encrypted_results_dir
    else:
        training_losses = np.load(unencrypted_numpy_dir + "unencrypted_training_losses.npy")
        training_accuracies = np.load(unencrypted_numpy_dir + "unencrypted_training_accuracies.npy")
        save_dir = unencrypted_results_dir

    # Loss Plotting:
    plt.title("Training Loss:")
    plt.ylabel("Loss")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_losses)
    plt.savefig(save_dir + "training_loss.png")
    plt.show()
    # Accuracy Plotting:
    plt.title("Training Accuracy:")
    plt.ylabel("Accuracy %")
    plt.xlabel("Epoch Iteration")
    plt.plot(training_accuracies, "g")
    plt.savefig(save_dir + "training_accuracy.png")
    plt.show()
