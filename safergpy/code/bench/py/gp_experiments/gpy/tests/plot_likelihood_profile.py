import matplotlib

# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
import matplotlib.pyplot as plt

import os
import sys
import pandas as pd
import numpy as np
import GPy
currentdir = os.getcwd()
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import libs.utils.gpy_plotting_lib as gpy_plotting_lib
from paramz.transformations import Exponent

try:
    data = pd.read_csv('/Users/sebastien/git-repos/test-functions/data/doe/branin_non_uniform_10d.csv', index_col=0)
except FileNotFoundError:
    data = pd.read_csv('/media/subhasish/Professional/L2S/gitlab_repos/test-functions/data/doe/branin_non_uniform_10d.csv', index_col=0)

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

# plt.title("ProfiledLikPath")

plt.ylabel("NLL")

#plt.legend(loc = "upper center")

plt.show()

# plt.savefig("/Users/sebastien/cache_matplotlib/plot.pgf")
# plt.savefig("/Users/sebastien/git-repos/gpstat-param/aistats2021/figures/Figure_5.pgf")
