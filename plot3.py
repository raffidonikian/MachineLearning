import matplotlib.pyplot as plt
import numpy as np

from starter import *


def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    X = X / 100
    Y = Y / 100
    X_test = X_test / 100
    Y_test = Y_test / 100
    model = Model(X.shape[1])
    layers = num_layers
    nodes = 450

    for i in range(layers):
        if activation is "ReLU":
            model.addLayer(DenseLayer(nodes, ReLUActivation()))
        else:
            model.addLayer(DenseLayer(nodes, TanhActivation()))
    model.addLayer(DenseLayer(2,LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X, Y, 1000, GDOptimizer(eta=.000001))
    Y_pred = model.predict(X_test)
    return np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))



#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_layerss = [1, 2, 3, 4]
mses = np.zeros((len(num_layerss), 2))

# for s in range(replicates):
sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_layers in enumerate(num_layerss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} layers done...'.format(num_layers))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_layerss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of layers')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_layers.png')
