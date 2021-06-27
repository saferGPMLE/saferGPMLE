import os, sys
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pandas as pd
import GPy
import libs.utils.gpy_plotting_lib as gpy_plotting_lib
import math
import numpy as np
from paramz.transformations import Exponent

n = 100
# gtol = None
# bfgs_factor = None

mean_value = 1210.55

variance_value = 2277543.94

optimum = np.array([27.04301504, 83.37540132])


try:
    data = pd.read_csv('/Users/sebastien/git-repos/test-functions/data/doe/branin_non_uniform_10d.csv', index_col=0)
except FileNotFoundError:
    data = pd.read_csv('/media/subhasish/Professional/L2S/gitlab_repos/test-functions/data/doe/branin_non_uniform_10d.csv', index_col=0)

x_train = data[['x0', 'x1']].values
z_train = data[['f_1']].values

mean_function = GPy.mappings.constant.Constant(input_dim=x_train.shape[1], output_dim=1, value=0.0)
kernel_function = GPy.kern.Matern52(input_dim=x_train.shape[1], variance=1, lengthscale=1, ARD=True)

model = GPy.models.GPRegression(x_train, z_train, kernel=kernel_function,
                                Y_metadata=None, normalizer=None,
                                noise_var=0, mean_function=mean_function)

model.Gaussian_noise.variance.fix()
# model.kern.lengthscale.constrain(Exponent())
# model.kern.variance.constrain(Exponent())

data_optim = gpy_plotting_lib.plot_multistart_optimization(model, n=n,
                                                           mean_value=mean_value, variance_value=variance_value,
                                                           optimum=optimum.copy(), init_type='profiled')


data_optim['dist_from_opt'] = data_optim.apply(
    lambda x: math.sqrt((x['rho1'] - optimum[0])**2 + (x['rho2'] - optimum[1])**2), axis=1)

bins = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

for thresold in bins:
    print("bin : {}, proportion : {}".format(thresold, (data_optim['dist_from_opt'] < thresold).mean()))
