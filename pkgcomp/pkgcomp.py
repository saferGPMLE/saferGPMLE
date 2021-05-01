import numpy as np
import pandas as pd
import argparse
from pythongp import pythongp_init

all_packages = ['scikit-learn', 'gpy', 'gpytorch', 'gpflow', 'openturns']

# Parse input
parser = argparse.ArgumentParser()
parser.add_argument('--pkg', choices=all_packages + ['all'], default='all',
                    help='Choose one package to be used, or all of them')
parser.add_argument('--n_train', default='50', type=int)
parser.add_argument('--filename', default='branin_uniform.csv')
args = parser.parse_args()

# List of packages to be tested
if args.pkg == 'all':
    package_list = all_packages
else:
    package_list = [args.pkg]

# Import data
filename = args.filename
print('filename =', filename)
data = pd.read_csv(filename)

# Train/test split
n_train = args.n_train
print('n_train =', n_train)


######################################################
#   A function to compute some metrics               #
#      for a given package, using a given data set   #
######################################################

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


####################################################
#   Compute metrics for all packages in the list   #
####################################################

np.random.seed(0)

for pkg in package_list:
    print("\n\nPython toolbox: ", pkg)
    pgp = pythongp_init.select_package(pkg)
    compute_metrics(pgp, data, n_train)
