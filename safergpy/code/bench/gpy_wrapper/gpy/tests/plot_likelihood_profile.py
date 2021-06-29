
import matplotlib.pyplot as plt

import sys
import pandas as pd
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

plt.figure(figsize=(0.75*6.4, 0.75*4.8))
plt.tight_layout()

gpy_plotting_lib.plot_paramz_likelihood_path(model, 'InvSoftPlus')

model.kern.lengthscale.constrain(Exponent())
model.kern.variance.constrain(Exponent())
gpy_plotting_lib.plot_paramz_likelihood_path(model, 'Log')

plt.axvline(x=0, color='b', label='Profiled init')
plt.axvline(x=1, color='k', label='Optimum')
plt.ylabel("NLL")
plt.show()

# plt.savefig("./Figure_5.pgf")
