from gp_experiments.gpy.libs.models.RationalMaternZeroMean import RationalMaternZeroMean
from gp_experiments.gpy.libs.models.CustomGPy import CustomGPy
from gp_experiments.gpy.libs.utils.metrics_computations import get_scaled_mse
import matplotlib.pyplot as plt
import numpy as np
import math
import sampling.sample

true_p = 0
true_variance = 1
true_rhos = 1

n = 100

#x = np.load('/Users/sebastien/git-repos/gp_param_exp/does/doe_10_1_unif.npy')
#x = np.linspace(0, 1, 10).reshape(-1, 1)

x = np.load('/Users/sebastien/git-repos/gp_param_exp/does/doe_10_3_unif.npy')

np.random.seed(0)

y = sampling.sample.get_gp_trajectories(x, true_p, 10, true_rhos, 1).T

y = 50 * y - 5

x = 3 * x - 4

# criteria = 'mle'
#
# print("start")
# model_trained = RationalMaternZeroMean(
#     p=0,
#     criteria=criteria,
#     untrained_lengthscale = 1,
#     untrained_variance = 5,
#     optim_scheme = [[10, 3.0], [1, 1.0]],
#     init = 'classic',
#     upper_lengthscale={0: 2},
#     fix_var=False
# )
#
# model_trained.set_data(x, y)
#
# model_trained.initialize()
#
# print(model_trained)
# print(model_trained.get_c_mean())
# print(model_trained.get_c_var())
# print(model_trained.get_ls())
# print(model_trained.get_objective_value())
#
#
# model_trained.train([0])
#
# print(model_trained)
# print(model_trained.get_c_mean())
# print(model_trained.get_c_var())
# print(model_trained.get_ls())
# print(model_trained.get_objective_value())
#
#
# print(
#     ( ( (model_trained._output_values - model_trained.get_loo_mean() ) ** 2) / model_trained.get_loo_var() ).mean()
# )
#
# print(model_trained.model._objective_grads(model_trained.model.optimizer_array))

#############################

# model_trained = CustomGPy(postvar_options = {"value": 0, "type": 'None'})
# model_trained.set_data(x, y)
#
# model_trained.initialize()
#
# print(x.min(0))
# print(x.max(0) - x.min(0))
# print(y.mean())
# print(y.std())
# print(model_trained._input_ordinates)
# print(model_trained._input_factors)
# print(model_trained._output_ordinate)
# print(model_trained._output_factor)
# print(model_trained._input_values - (x - model_trained._input_ordinates)/model_trained._input_factors)
# print(model_trained._output_values - (y - model_trained._output_ordinate)/model_trained._output_factor)
# print(model_trained.get_c_var())
# print(model_trained.get_c_mean())
# print(model_trained.get_ls())
# print(model_trained.get_objective_value())
# print("")
#
# x_new = np.load('/Users/sebastien/git-repos/gp_param_exp/does/doe_20_3_unif.npy')
# y_new = sampling.sample.get_gp_trajectories(x_new, true_p, true_variance, true_rhos, 1).T
#
# model_trained.clean()
# model_trained.set_data(x_new, y_new)
#
# print(x_new.min(0))
# print(x_new.max(0) - x_new.min(0))
# print(y_new.mean())
# print(y_new.std())
# print(model_trained._input_ordinates)
# print(model_trained._input_factors)
# print(model_trained._output_ordinate)
# print(model_trained._output_factor)
# print(model_trained._input_values - (x_new - model_trained._input_ordinates)/model_trained._input_factors)
# print(model_trained._output_values - (y_new - model_trained._output_ordinate)/model_trained._output_factor)
# print(model_trained.get_c_var())
# print(model_trained.get_c_mean())
# print(model_trained.get_ls())
# print(model_trained.get_objective_value())
# print("")


#############################
#
# model_trained = RationalMaternZeroMean(
#     p=0,
#     criteria='mle',
#     untrained_lengthscale = 1,
#     untrained_variance = 5,
#     optim_scheme = [[10, 3.0], [1, 1.0]],
#     init = 'classic',
#     upper_lengthscale={0: 2},
#     fix_var=False
# )
#
# model_trained.set_data(x, y)
#
# model_trained.initialize()
#
# print(x.min(0))
# print(x.max(0) - x.min(0))
# print(y.mean())
# print(y.std())
# print(model_trained._input_ordinates)
# print(model_trained._input_factors)
# print(model_trained._output_ordinate)
# print(model_trained._output_factor)
# print(model_trained._input_values - (x - model_trained._input_ordinates)/model_trained._input_factors)
# print(model_trained._output_values - (y - model_trained._output_ordinate)/model_trained._output_factor)
# print("")
#
# x_new = np.load('/Users/sebastien/git-repos/gp_param_exp/does/doe_20_3_unif.npy')
# y_new = sampling.sample.get_gp_trajectories(x_new, true_p, true_variance, true_rhos, 1).T
#
# model_trained.set_data(x_new, y_new)
#
# print(x_new.min(0))
# print(x_new.max(0) - x_new.min(0))
# print(y_new.mean())
# print(y_new.std())
# print(model_trained._input_ordinates)
# print(model_trained._input_factors)
# print(model_trained._output_ordinate)
# print(model_trained._output_factor)
# print(model_trained._input_values - (x_new - model_trained._input_ordinates)/model_trained._input_factors)
# print(model_trained._output_values - (y_new - model_trained._output_ordinate)/model_trained._output_factor)
# print("")


###############################

print("start")

model_trained = CustomGPy(postvar_options={"value": 0, "type": 'None'})
model_trained.set_data(x, y)

model_trained.initialize()

print(model_trained)

print(model_trained.get_c_mean())
print(model_trained.get_c_var())
print(model_trained.get_ls())
print(model_trained.get_objective_value())


###############################

# model_trained.set_loo_posterior_metrics()
# mu_analytic = model_trained.get_loo_mean().copy()
# var_analytic = model_trained.get_loo_var().copy()
#
# mu_loo_empirical = []
# var_loo_empirical = []
#
# for i in range(x.shape[0]):
#     model_trained.set_data(np.delete(x, i, 0), np.delete(y, i, 0))
#
#     predictions = model_trained.predict(x[[i], :])
#     mu_loo_empirical.append(predictions[0][0, 0])
#     var_loo_empirical.append(predictions[1][0, 0])
#
# mu_loo_empirical = np.array(mu_loo_empirical)
# var_loo_empirical = np.array(var_loo_empirical)
#
# print("")
# print(mu_loo_empirical.__repr__())
# print( ( (mu_loo_empirical.reshape(-1) - model_trained.transform_post_mean(mu_analytic).reshape(-1))**2 ).sum() )
#
# print(var_loo_empirical.__repr__())
# print( ( (var_loo_empirical.reshape(-1) - model_trained.transform_post_var(var_analytic).reshape(-1))**2 ).sum() )

#############################

x = np.linspace(0, 1, 5).reshape(-1, 1)

np.random.seed(0)

y = sampling.sample.get_gp_trajectories(x, true_p, 10, true_rhos, 1).T

y = 3 * y - 50

x = 3 * x - 4

model_trained = CustomGPy(postvar_options={"value": 0, "type": 'truncate'}, init='classic_profiled')
model_trained.set_data(x, y)

#############################

x_new = np.linspace(x.min(), x.max(), 10).reshape(-1, 1)
y_new = sampling.sample.get_gp_trajectories(x_new, true_p, true_variance, true_rhos, 1).T

model_trained.set_data(x_new, y_new)

x_new_new = np.linspace(x.min(), x.max(), 1000).reshape(-1, 1)

plt.plot(x_new, y_new, 'o')
plt.plot(x_new_new, model_trained.predict(x_new_new)[0], label='post mean')
plt.plot(x_new_new, model_trained.predict(x_new_new)[0] - 1.96 * np.sqrt(model_trained.predict(x_new_new)[1]), label='lower CI')
plt.plot(x_new_new, model_trained.predict(x_new_new)[0] + 1.96 * np.sqrt(model_trained.predict(x_new_new)[1]), label='upper CI')

for i in range(3):
    plt.plot(x_new_new, model_trained.sample_y(x_new_new, 1).reshape(-1), color='k')

plt.plot(x_new_new, model_trained.get_y_quantiles(0.9999, x_new_new), label='quantile')
plt.plot(x_new_new, model_trained.get_y_quantiles(0.0001, x_new_new), label='quantile')
plt.legend()

##########################

model_trained.train()

np.random.seed(0)
#x_new = np.random.uniform(size = [10, 5])
x_new = np.linspace(x.min(), x.max(), 10000).reshape(-1, 1)


print(model_trained)
print(model_trained.get_c_mean())
print(model_trained.get_c_var())
print(model_trained.get_ls())

print(model_trained.predict(x_new)[0])
print(model_trained.predict(x_new)[1])

print(model_trained.get_objective_value())

plt.plot(x, y, 'o')
plt.plot(x_new, model_trained.predict(x_new)[0])
plt.plot(x_new, model_trained.predict(x_new)[0] - 3 * np.sqrt(model_trained.predict(x_new)[1]))
plt.plot(x_new, model_trained.predict(x_new)[0] + 3 * np.sqrt(model_trained.predict(x_new)[1]))


# model_trained.train()

#print(model_trained)
#print(model_trained.model.optimizer_array)
from gp_experiments.gpy.libs.utils.gpy_estimation_lib import analytical_zero_mean_variance_optimization as prof

criteria = 'mse_loo'

print("start")
model_trained = RationalMaternZeroMean(
    p=0,
    criteria=criteria,
    untrained_lengthscale=1,
    untrained_variance=1,
    optim_scheme=[[10, 3.0], [1, 1.0]],
    init='classic',
    upper_lengthscale={0: 2},
    fix_var=False
)

model_trained.set_data(x, y)

model_trained.initialize()

model_trained.train([0])

mu_analytic = model_trained.get_loo_mean().copy()
var_analytic = model_trained.get_loo_var().copy()

mu_loo_empirical = []
var_loo_empirical = []

for i in range(x.shape[0]):
    model_trained.set_data(np.delete(x, i, 0), np.delete(y, i, 0))

    predictions = model_trained.predict(x[[i], :])
    mu_loo_empirical.append(predictions[0][0, 0])
    var_loo_empirical.append(predictions[1][0, 0])

mu_loo_empirical = np.array(mu_loo_empirical)
var_loo_empirical = np.array(var_loo_empirical)


####################


model_trained.set_loo_posterior_metrics()

get_scaled_mse(mu_pred=model_trained.get_loo_mean().reshape(-1),
                   var_pred=model_trained.get_loo_var().reshape(-1),
                   y=y.reshape(-1))

#print(model_trained)
print(y.mean())
print(y.var())
print(x.std(0))

#model.objective_function_gradients()
#model._grads(model.optimizer_array)
#model.checkgrad(verbose = True, tolerance = 10**(-8))

#eps = 0.0001
#
# grad = model._grads(model.optimizer_array)
# num_grad = []
# x_model = model.optimizer_array.copy()
#
# for i in range(4):
#     x1 = x_model.copy()
#     x1[i] = x1[i] + eps
#     f_2 = model._objective(x1)
#     x1[i] = x1[i] - 2*eps
#     f_1 = model._objective(x1)
#     num_grad.append((f_2 - f_1)/(2*eps))

model_trained.train_with_fixed_nu()

print(model_trained)
print(model_trained.model.optimizer_array)

model_trained.train([0, 1, 2, 3])

print(model_trained)
print(model_trained.model.optimizer_array)

############################################

import matplotlib.pyplot as plt

p_0 = 5

p_s = list(range(0, 20))

values = [(2*p + 3)**2 / (4*(p-p_0) + 1) for p in p_s]
#values = [(2*p + 3)**2 / (4*p-2*p_0 + 3) for p in p_s]

plt.plot(p_s, values)
