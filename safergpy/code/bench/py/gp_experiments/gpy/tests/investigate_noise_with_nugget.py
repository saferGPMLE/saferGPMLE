import os
import sys
import pandas as pd
import numpy as np
import GPy
import libs.utils.gpy_plotting_lib as gpy_plotting_lib
from paramz.transformations import Exponent

path = sys.argv[1]

data = pd.read_csv(path, index_col=0)

x_train = data[[c for c in data.columns if 'x' in c]].values
z_train = data[['f_1']].values

mean_function = GPy.mappings.constant.Constant(input_dim=x_train.shape[1], output_dim=1, value=0.0)
kernel_function = GPy.kern.Matern52(input_dim=x_train.shape[1], variance=1, lengthscale=1, ARD=True)

model = GPy.models.GPRegression(x_train, z_train, kernel=kernel_function,
                                Y_metadata=None, normalizer=None,
                                noise_var=0, mean_function=mean_function)

model.Gaussian_noise.variance.fix()
model.kern.lengthscale.constrain(Exponent())
model.kern.variance.constrain(Exponent())


########################################################################################

# For general datasets
model.Mat52.variance = 2437888.1269414485
model.constmap.C = 149.89436419735705
model.Mat52.lengthscale = [27.39835089, 83.54421333]

model.Gaussian_noise.variance = (10 ** (-2)) * model.Mat52.variance.copy()

Ky = model.posterior._K.copy() + np.diag(z_train.shape[0] * [model.Gaussian_noise.variance.copy()[0]])

print("error : {}".format(np.sqrt(((model.predict(x_train)[0] - z_train) ** 2).mean())/z_train.std()))

print("NLL : {}".format(model.objective_function()))

########################################################################################

lambdas = np.linalg.eig(Ky)[0]

print("condition number log_det : 10^({})".format(np.log10(np.sqrt((1/lambdas**2).sum() * (lambdas**2).sum())/np.abs(np.log((lambdas).sum())))))
print("condition number : 10^({})".format(np.log10(np.linalg.cond(Ky))))

idx_param = 2

gpy_plotting_lib.plot_taylor(model, idx_param, diagonalize=True, width=1e-3, n=1000)
gpy_plotting_lib.decompose_all(model, idx_param, diagonalize=True, width=1e-10, n=10000)
