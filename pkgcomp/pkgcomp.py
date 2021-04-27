import numpy as np
import pandas as pd
from pythongp import pythongp_init

np.random.seed(0)

# Import data
data = pd.read_csv('branin_uniform.csv')

# List of libraries to be tested
libraries = ['Sklearn', 'GPy', 'GPytorch', 'GPflow', 'ot']

# Train/test split
n_train = 50

def compute_metrics(pgp, data, n_train):
    '''
    Fits GP on multivariate datasets
    Author: S.B
    Description: The following code implements Gaussian Process regression
    and computes a prediction metric EMRMSE for a python toolbox
    '''

    # Extract & load input data
    cols = [c for c in data.columns if 'x' in c]
    x_train, x_test =                \
        data[cols].values[:n_train], \
        data[cols].values[n_train:]

    # Extract output data
    z_train, z_test =                               \
        np.array(data[data.columns[-1]][:n_train]), \
        np.array(data[data.columns[-1]][n_train:])

    # Define kernel parameters
    kernel_param = {
        'name':                'Matern',
        'lengthscale':         [1, 1],
        'order':               2.5,
        'lengthscale_bounds':  '(1e-5, 1e5)',
        'scale':               1.0
    }

    # Contruct GP model
    pgp.load_data(x_train, z_train)
    pgp.set_kernel(kernel_param, ard=True)
    pgp.set_mean('zero')
    pgp.init_model(noise=0.0001)
    pgp.optimize(param_opt='MLE', itr=10)

    # Make predictions
    z_postmean, z_postvar = pgp.predict(x_test)

    # Compute EMRMSE
    emrmse = np.sqrt(np.mean(np.square(
        np.array(z_test) - np.array(z_postmean))))
    print("EMRMSE", emrmse)


# Compute metrics for all libraries
for lib in libraries:
    print("\n\nPython toolbox: ", lib)
    pgp = pythongp_init.select_library(lib)
    compute_metrics(pgp, data, n_train)

