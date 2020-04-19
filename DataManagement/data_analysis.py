from DataManagement.data_loading import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

def analyze_boston_housing():
    (x_train, y_train), (x_test, y_test) = load_boston_housing(100, 100)
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

    plt.show()



if __name__ == "__main__":
    analyze_boston_housing()