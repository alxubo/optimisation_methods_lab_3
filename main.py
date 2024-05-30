import time
import torch

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import scipy
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def plot_regression(X, y, y_pred=None, log=None, title="Linear Regression", x_title="", y_title=""):
    plt.figure(figsize=(16, 6))
    plt.rcParams['figure.dpi'] = 227
    plt.scatter(X, y, label='Data', c='#388fd8', s=6)
    if log is not None:
        for i in range(len(log)):
            plt.plot(X, log[i][0] * X + log[i][1], lw=1, c='#caa727', alpha=0.35)

    if y_pred is not None:
        plt.plot(X, y_pred, c='#ff7702', lw=3, label='Regression')

    plt.title(title)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.legend(frameon=True, loc=1, fontsize=10, borderpad=.6)
    plt.tick_params(direction='out', length=6, color='#a0a0a0', width=1, grid_alpha=.6)
    plt.show()


def scale(x):
    min_x = x.min()
    max_x = x.max()
    return pd.Series([(i - min_x) / (max_x - min_x) for i in x])


def sgd_linear_regression(x, y, lr=0.05, epoch=10, batch_size=1, lf_function_name=None, lr_decay_exp=None,
                          lr_decay_ladder=None,
                          max_batch=1, epsilon=10 ** -8):
    start_time = time.time()
    operations_amount = 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss

    m, b = 0.5, 0.5  # initial parameters
    log, mse = [], []  # lists to store learning process
    amout_of_iterations = epoch * (max_batch // batch_size)

    for i in range(epoch * (max_batch // batch_size)):
        indexes = np.random.randint(0, len(x), batch_size)  # random sample

        Xs = x.iloc[indexes]
        ys = y.iloc[indexes]
        N = len(Xs)

        f = ys - (m * Xs + b)

        # Updating parameters m and b
        old_m = m
        old_b = b

        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)

        operations_amount += 9

        if abs(old_m - m) < epsilon and abs(old_b - b) < epsilon:
            amout_of_iterations = i
            break

        if lf_function_name == "exponential":
            if lr_decay_exp is None:
                print("lr_decay_exp should not be None. It should be a float")
                return
            lr *= lr_decay_exp
            operations_amount += 1

        if lf_function_name == "ladderal":
            if lr_decay_ladder is None:
                print("lr_decay_ladder should not be None. It should be a list of floats")
                return
            if i in lr_decay_ladder:
                lr *= lr_decay_ladder[i]

        log.append((m, b))
        mse.append(mean_squared_error(y, m * x + b))

    return m, b, log, mse, time.time() - start_time, operations_amount, (process.memory_info().rss - ram_usage) / (
            1024 ** 2), amout_of_iterations


def sgd_linear_regression_keras(X, y, learning_rate=0.01, epochs=10, batch_size=5, function_name="SGD"):
    # Reshape X into a 2D array
    X_reshaped = X.values.reshape(-1, 1)  # Assuming X is a pandas Series

    # Scale the features and target variables
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))  # Assuming y is also a pandas Series

    # Create a sequential model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1, input_shape=(X_scaled.shape[1],)))

    start_time = time.time()
    process = psutil.Process()
    ram_usage = process.memory_info().rss

    optimizer = None
    if function_name == "SGD":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)  # SGD
    elif function_name == "Nesterov":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)  # Nesterov optimizer
    elif function_name == "Momentum":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)  # Momentum optimizer
    elif function_name == "AdaGrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)  # AdaGrad optimizer
    elif function_name == "RMSProp":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)  # RMSProp optimizer
    elif function_name == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Adam optimizer

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Fit the model using SGD
    model.fit(X_scaled, y_scaled, epochs=epochs, batch_size=batch_size)

    # Get the trained weights
    theta_sgd = model.get_weights()[0][0, 0]
    intercept_sgd = model.get_weights()[1][0]

    return theta_sgd, intercept_sgd, time.time() - start_time, (process.memory_info().rss - ram_usage) / (1024 ** 2)


def get_data_set(name='california_housing'):
    if name == 'california_housing':
        dataset = pd.read_csv('./datasets/housing.csv')

        # Removing imperfections in the dataset
        dataset = dataset[dataset.median_house_value < 490000]
        dataset = dataset[dataset.median_income < 8]

        return scale(dataset.median_income), scale(dataset.median_house_value)

    elif name == 'wine':
        dataset = pd.read_csv('datasets/winequality-red.csv')

        # # Removing imperfections in the dataset
        # dataset = dataset[dataset.median_house_value < 490000]
        # dataset = dataset[dataset.median_income < 8]
        return scale(dataset.fixed_acidity), scale(dataset.volatile_acidity)

    elif name == 'sinosoidal_dataset':
        dataset = pd.read_csv('datasets/sinosoidal_dataset.csv')

        # # Removing imperfections in the dataset
        # dataset = dataset[dataset.median_house_value < 490000]
        # dataset = dataset[dataset.median_income < 8]
        return scale(dataset.x), scale(dataset.y)
    else:
        return


def run_all_keras_methods(x, y):
    methods = ["SGD", "Nesterov", "Momentum", "AdaGrad", "RMSProp", "Adam"]
    batches = [50]
    for batch in batches:
        result = []
        # for method in methods:
        m, b, time_used, ram_used = sgd_linear_regression_keras(x, y, batch_size=batch, function_name="RMSProp")

        y_pred = m * x + b
        # result.append(method + "&" + str(batch) + "&" + str(round(time_used, 5)) + "&" + str(
        #     round(mean_squared_error(y, y_pred), 8)) + "&" +
        #               str(round(ram_used, 5)) + "\\\\")

        plot_regression(x, y, y_pred, title="Linear Regression with SGD", x_title="Median Income",
                        y_title="House Price")
        # for st in result:
        #     print(st)


def create_sinosoidal_dataset(n_samples, n_features, amplitude=1.0, frequency=1.0, noise=0.1):
    # Create the features.
    features = np.linspace(0, 1, n_samples)

    # Create the labels.
    labels = amplitude * np.sin(2 * np.pi * frequency * features) + noise * np.random.randn(n_samples)

    # Return the features and labels.

    return features, labels


def sgd_l1_polynomial_regression(x, y, degree=1, lr=0.05, epoch=10, batch_size=1, lf_function_name=None,
                                 lr_decay_exp=None, lr_decay_ladder=None, max_batch=1, epsilon=10 ** -8):
    start_time = time.time()
    operations_amount = 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss

    # Initialize the coefficients
    coefficients = np.ones(degree + 1)

    # Lists to store learning process
    log, mse = [], []
    amout_of_iterations = epoch * (max_batch // batch_size)

    for i in range(epoch * (max_batch // batch_size)):
        indexes = np.random.randint(0, len(x), batch_size)  # random sample

        Xs = x.iloc[indexes]
        ys = y.iloc[indexes]
        N = len(Xs)

        X_powers = np.array([Xs ** i for i in range(degree + 1)]).T
        f = ys - np.dot(X_powers, coefficients)

        # Updating coefficients
        old_coefficients = coefficients.copy()

        for j in range(degree + 1):
            coefficients[j] -= lr * (-2 * np.dot(f, X_powers[:, j]).sum() / N + 2 * lr * coefficients[j] / N)

        operations_amount += (9 * (degree + 1))

        if np.linalg.norm(old_coefficients - coefficients) < epsilon:
            amout_of_iterations = i
            break

        if lf_function_name == "exponential":
            if lr_decay_exp is None:
                print("lr_decay_exp should not be None. It should be a float")
                return
            lr *= lr_decay_exp
            operations_amount += 1

        if lf_function_name == "ladderal":
            if lr_decay_ladder is None:
                print("lr_decay_ladder should not be None. It should be a list of floats")
                return
            if i in lr_decay_ladder:
                lr *= lr_decay_ladder[i]

        log.append(coefficients)
        mse.append(mean_squared_error(y, np.dot(X_powers, coefficients)))

    return coefficients, mse, log, operations_amount, time.time() - start_time, amout_of_iterations, ram_usage


def main():
    x, y = get_data_set()

    plot_regression(x, y, title='California dreaming')

    m, b, log, mse, time_used, arifmetic_operations, ram_used, amount_of_iterations = sgd_linear_regression(x, y,
                                                                                                            lr=0.1,
                                                                                                            epoch=5,
                                                                                                            batch_size=500,
                                                                                                            # lf_function_name="exponential",
                                                                                                            # lr_decay_exp=0.95,
                                                                                                            max_batch=19364)
    y_pred = m * x + b

    print("MSE:", round(mean_squared_error(y, y_pred), 8))
    print("time_used: ", round(time_used, 5))
    print("arifmetic_operations: ", arifmetic_operations)
    print("ram_used in megabytes: ", round(ram_used, 5))

    plot_regression(x, y, y_pred, log=log, title="Linear Regression with SGD", x_title="Fixed acidity",
                    y_title="Volatile acidity")

    m, b, time_used, ram_used = sgd_linear_regression_keras(x, y, function_name="RMSProp")
    y_pred = m * x + b

    print("time_used: ", round(time_used, 5))
    print("ram_used in megabytes: ", round(ram_used, 5))

    print("MSE:", round(mean_squared_error(y, y_pred), 5))
    plot_regression(x, y, y_pred, title="Linear Regression with SGD", x_title="Fixed acidity",
                    y_title="Volatile acidity")
    run_all_keras_methods(x, y)


if '__main__' == __name__:
    main()
