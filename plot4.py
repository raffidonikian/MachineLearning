import numpy as np
import matplotlib.pyplot as plt

from starter import *
from plot1 import *


def neural_network(X, Y, Xs_test, Ys_test):
    X = X / 100
    Y = Y / 100

    model = Model(X.shape[1])
    layers = 3
    nodes = 450
    for i in range(layers):
        model.addLayer(DenseLayer(nodes, ReLUActivation()))
    model.addLayer(DenseLayer(2,LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X,Y,100,GDOptimizer(eta=.000001))

    mses = []
    for i in range(len(Xs_test)):
        X_test = Xs_test[i]
        Y_test = Ys_test[i]
        X_test = X_test / 100
        Y_test = Y_test / 100
        Y_pred = model.predict(X_test)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(10, 310, 20)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression:
            mse = linear_regression(X, Y, Xs_test, Ys_test)
            mses[t, s, 0] = mse

            ### Second-order Polynomial regression:
            mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse

            ### 3rd-order Polynomial regression:
            mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 3] = mse

            ### Generative model:
            mse = generative_model(X, Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse

            ### Oracle model:
            mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    main()
