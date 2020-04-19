import matplotlib.pyplot as plt
import numpy as np

def scaled_squared_complexity_plots():
    x_iter = []
    for i in range(-50, 50):
        x_iter.append(i / 10)
    x_iter = np.array(x_iter)
    y_values = np.multiply(x_iter, x_iter)

    c1 = 0.1
    c2 = 0.01
    c3 = 0.001
    c4 = 0.0001

    plt.plot(x_iter, y_values, label="c = 1")
    plt.plot(x_iter, y_values * c1, label="c = 0.1")
    plt.plot(x_iter, y_values * c2, label="c = 0.01")
    plt.plot(x_iter, y_values * c3, label="c = 0.001")
    plt.plot(x_iter, y_values * c4, label="c = 0.0001")
    plt.legend(loc="upper left")
    plt.title("Complexity Factor Plot of [1, 0.1, 0.01, 0.001, 0.0001]")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    scaled_squared_complexity_plots()