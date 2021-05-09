import platform
import argparse
import numpy as np
import pandas as pd
from pythongp import pythongp_init

# Parse input
parser = argparse.ArgumentParser()
parser.add_argument('--pkg', default='gpy')
parser.add_argument('--n_train', default='50', type=int)
parser.add_argument('--filename', default='branin_uniform.csv')
parser.add_argument('--output', default='')
args = parser.parse_args()

# Package to be tested
pkg_str = args.pkg
c = pkg_str.split('==')
if len(c) == 1:
    pkg = pkg_str
    pkg_ver = None
elif len(c) == 2:
    pkg = c[0]
    pkg_ver = c[1]
else:
    raise Exception("Incorrect package name")

# TODO: check that the correct package version is installed

# Import data
filename = args.filename
print('filename =', filename)
data = pd.read_csv(filename)

# Train/test split
n_train = args.n_train
print('n_train =', n_train)

# Output file
if not args.output:
    output = None
else:
    output = open(args.output, "a")

# OS info
os_info = platform.system() + '-' + platform.release()


######################################################
#   A function to compute some metrics               #
#      for a given package, using a given data set   #
######################################################

def compute_metrics(pgp, data, n_train, output):
    '''
    Fits GP on multivariate datasets
    Author: S.B
    Description: The following code implements Gaussian Process regression
    and computes a prediction metric ERMSPE for a python toolbox
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
        'variance':            1.0
    }

    # Contruct GP model
    pgp.load_data(x_train, z_train)
    pgp.set_kernel(kernel_param, ard=True)
    pgp.set_mean('zero')
    pgp.init_model(noise=0.0001)
    pgp.optimize(param_opt='MLE', itr=10)

    # Extract negative log-likelihood
    NLL = pgp.get_NLL()

    # Compute ERMSPE
    z_postmean, z_postvar = pgp.predict(x_test)
    ermspe = np.sqrt(np.mean(np.square(
        np.array(z_test) - np.array(z_postmean))))

    print("\n")
    print("[compute_metrics]  RESULTS:")
    print("[compute_metrics]  | NLL    =", NLL)
    print("[compute_metrics]  | EMRMSPE =", ermspe)

    if output:
        output.write("{:.16e},".format(NLL))
        output.write("{:.16e}\n".format(ermspe))


####################################################
#   Compute metrics for all packages in the list   #
####################################################

np.random.seed(0)

print("\n\nPython toolbox: ", pkg_str)

if output:
    output.write(pkg_str + ",")
    output.write(os_info + ",")

pgp = pythongp_init.select_package(pkg)
compute_metrics(pgp, data, n_train, output)

if output:
    output.close()
