from DataManagement.data_loading import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

def analyze_boston_housing():
    (x_train, y_train), (x_test, y_test) = load_boston_housing(1)
    print(f"Length of Training Data = {x_train.shape[0]}")
    print(f"Length of Testing Data = {x_test.shape[0]}")
    print(f"Number of Data Features = {x_train.shape[1]}")

    boston_housing = load_boston()
    # print(boston_housing.DESCR)

    y_total = np.vstack((y_train, y_test))

    print(f"Max target value = {max(y_total)[0]}")
    print(f"Min target value = {min(y_total)[0]}")
    print(f"Mean target value = {np.average(y_total)}")

    plt.plot(y_test, "m")
    plt.xlabel("Data Index")
    plt.ylabel("Housing Price in 1000's of USD")
    plt.title("Boston Housing Regression Testing Targets")
    plt.grid()

    plt.show()
    return


def analyze_year_prediction():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_year_prediction()

    print(f"Total Dataset Length = {x_train.shape[0] + x_valid.shape[0] + x_test.shape[0]}")
    print(f"Length of Training Data = {x_train.shape[0]}")
    print(f"Length of Validation Data = {x_valid.shape[0]}")
    print(f"Length of Testing Data = {x_test.shape[0]}")
    print(f"Number of Data Features = {x_train.shape[1]}")
    y_total = np.vstack((y_test, np.vstack((y_train, y_valid))))
    print(f"Mean target value = {np.average(y_total)}")
    print(f"Median target value = {np.median(y_total)}")

    print("x Examples", np.round(x_train[14, :7], 2))
    print("y Examples", y_train[10:15])


    plt.plot(y_test[100:300], "b")
    plt.xlabel("Data Index")
    plt.ylabel("Year Song was Released")
    plt.title("Year Prediction Regression Testing Targets")
    plt.grid()
    plt.show()


    return


def analyze_mnist_dataset():
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_MNIST()

    print(f"Total Dataset Length = {x_train.shape[0] + x_valid.shape[0] + x_test.shape[0]}")
    print(f"Length of Training Data = {x_train.shape[0]}")
    print(f"Length of Validation Data = {x_valid.shape[0]}")
    print(f"Length of Testing Data = {x_test.shape[0]}")
    print(f"Number of Data Features = {x_train.shape[1]}")
    y_total = np.vstack((y_test, np.vstack((y_train, y_valid))))
    print(f"Mean target value = {np.average(y_total)}")
    print(f"Median target value = {np.median(y_total)}")
    print(0.15 * 31000)
    print(0.7*31000)
    i = 22
    i_sample = x_train[i].reshape((28, 28))
    plt.imshow(i_sample)
    plt.title("Dataset Input with Label 9")
    plt.set_cmap("Greys")
    plt.show()
    return


if __name__ == "__main__":
    analyze_mnist_dataset()