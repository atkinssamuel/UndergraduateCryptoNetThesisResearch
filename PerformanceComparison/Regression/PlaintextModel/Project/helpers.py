import matplotlib.pyplot as plt


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


def plot_regression_outputs(targets, predictions, title, x_label, y_label, save_directory="", save_title="", show=False):
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