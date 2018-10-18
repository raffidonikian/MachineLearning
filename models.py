import numpy as np
import scipy.spatial
from starter import *
import sklearn.preprocessing


#####################################################################
## Models used for predictions.
#####################################################################
def compute_update(single_obj_loc, sensor_loc, single_distance):
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def linear_regression(X, Y, Xs_test, Ys_test):
    w = np.linalg.solve(X.T @ X, X.T @ Y)
    mses = []
    for i in range(len(Xs_test)):
        X_test = Xs_test[i]
        Y_test = Ys_test[i]
        Y_pred = X_test @ w
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def poly_regression_second(X, Y, Xs_test, Ys_test):
    poly2 = sklearn.preprocessing.PolynomialFeatures(2)
    X = poly2.fit_transform(X)
    w = np.linalg.solve(X.T @ X, X.T @ Y)
    mses = []

    for i in range(len(Xs_test)):
        X_test = poly2.fit_transform(Xs_test[i])
        Y_test = Ys_test[i]
        Y_pred = X_test @ w
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses



def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    poly3 = sklearn.preprocessing.PolynomialFeatures(3)
    X = poly3.fit_transform(X)
    w = np.linalg.solve(X.T @ X, X.T @ Y)
    mses = []

    for i in range(len(Xs_test)):
        X_test = poly3.fit_transform(Xs_test[i])
        Y_test = Ys_test[i]
        Y_pred = X_test @ w
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def neural_network(X, Y, Xs_test, Ys_test):
    model = Model(X.shape[1])
    layers = 2
    nodes = 100
    for i in range(layers):
        model.addLayer(DenseLayer(nodes, ReLUActivation()))
    model.addLayer(DenseLayer(2,LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X,Y,500,GDOptimizer(eta=.000001))

    mses = []
    for i in range(len(Xs_test)):
        X_test = Xs_test[i]
        Y_test = Ys_test[i]
        Y_pred = model.predict(X_test)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses
