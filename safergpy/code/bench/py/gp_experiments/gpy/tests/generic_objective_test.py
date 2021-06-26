
import numpy as np
import time
import GPy
import gp_experiments.gpy.libs.utils.metrics_computations
import os
import sys


def get_trajectories(x, true_p, true_variance, true_rho, N_traj):
    kernel_function = GPy.kern.RationalMatern(p=true_p, input_dim=x.shape[1],
                                        variance=true_variance,
                                        lengthscale=true_rho,
                                        ARD=True)

    true_K = kernel_function.K(x)

    y = np.random.multivariate_normal(np.zeros([x.shape[0]]), true_K, size=N_traj)

    return y

x = np.array(np.load(os.path.join(os.getenv("HOME"), 'git-repos', 'gp_param_exp',
                             'does', 'doe_10_3_unif.npy')))

true_p= 2
true_variance = 1
true_rhos = x.shape[1] * [1]
N_traj = 300
N_simulations = 100

y = get_trajectories(x, true_p, true_variance, true_rhos, N_traj)

y = y[[0], :].reshape(-1, 1)

from gp_experiments.gpy.libs.models.RationalMaternZeroMean import RationalMaternZeroMean

p = true_p
variance = 0.78
rho = np.random.uniform(size=x.shape[1]) + 1

criteria = sys.argv[1]

model = RationalMaternZeroMean(p=p, criteria=criteria, init='custom', fix_noise=True,
                                   untrained_variance=variance,
                                   untrained_lengthscale=rho.copy())

model.set_data(x, y)

print(model.model._objective_grads(model.model.optimizer_array))

print(model.model.checkgrad(verbose=True, tolerance=10**(-8)))


def compute_mse(x, y):
    assert x.shape == y.shape, 'Shape issue'
    return ((x - y) ** 2).mean() / (x ** 2).mean()


def get_loo(ref_model, x, y):
    model = RationalMaternZeroMean(p=ref_model.p, criteria=ref_model.criteria, fix_noise=True,
                                   untrained_variance=ref_model.model.kern.variance.copy(),
                                   untrained_lengthscale=ref_model.model.kern.lengthscale.copy(), init='custom')

    mu_loo_empirical = []
    var_loo_empirical = []

    for i in range(x.shape[0]):
        model.set_data(np.delete(x, i, 0), np.delete(y, i, 0))

        predictions = model.predict(x[[i], :])
        mu_loo_empirical.append(predictions[0][0, 0])
        var_loo_empirical.append(predictions[1][0, 0])

    mu_loo_empirical = np.array(mu_loo_empirical)
    var_loo_empirical = np.array(var_loo_empirical)

    return mu_loo_empirical, var_loo_empirical

##################################################

test_kernel_function = GPy.kern.RationalMatern(p=model.p, input_dim=x.shape[1],
                                variance=model.model.kern.variance.copy(),
                                lengthscale=model.model.kern.lengthscale.copy(),
                                ARD=True)

test_model = GPy.models.GPRegression(x, y, kernel=test_kernel_function,
                                            Y_metadata=None, normalizer=None,
                                            noise_var=0, mean_function=None)

##################################################

x_test = np.random.uniform(size=(1000, x.shape[1]))

predictions = model.predict(x_test)

test_predictions = test_model.predict(x_test)


print("Predictions check")
print( compute_mse(predictions[0], test_predictions[0]) )
print( compute_mse(predictions[1], test_predictions[1]) )

####################################################

model.set_loo_posterior_metrics()

mu_loo_analytical = model.get_loo_mean().reshape(-1)
var_loo_analytical = model.get_loo_var().reshape(-1)

if criteria not in ['mle', 'gcv', 'kernel_alignment']:
    print("Analytical check")
    print(compute_mse(mu_loo_analytical, model.model.L_K.mu_loo))
    print(compute_mse(var_loo_analytical, model.model.L_K.var_loo))

mu_loo_empirical, var_loo_empirical = get_loo(model, x, y)

print("Numeric vs analytical")
print(compute_mse(mu_loo_analytical, mu_loo_empirical))
print(compute_mse(var_loo_analytical, var_loo_empirical))

##############################################################

if criteria == 'mse_loo':

    #model.set_data(x, y)

    print("LOO test objective")

    print(model.model.objective_function())
    print(((mu_loo_empirical.reshape(-1) - y.reshape(-1)) ** 2).mean())

    ###############################################################

    print("LOO test optimize")

    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    mu_loo_empirical, var_loo_empirical = get_loo(model, x, y)

    print(model.model.objective_function())
    print(((mu_loo_empirical.reshape(-1) - y.reshape(-1)) ** 2).mean())

#################################################################

if criteria == 'kernel_alignment':

    import gp_experiments.gpy.libs.utils.metrics_computations

    print("KA")
    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_kernel_alignment(model.model.posterior._K, model.model.Y))

    print("optimize KA")
    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    print("KA")
    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_kernel_alignment(model.model.posterior._K, model.model.Y))

#################################################################

if criteria == 'gcv':

    import gp_experiments.gpy.libs.utils.metrics_computations

    print("GCV")
    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_gcv_test(model.model.posterior._K,
                                                                                  model.model.Y)[0, 0])

    print("optimize GCV")

    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    print("GCV")
    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_gcv_test(model.model.posterior._K,
                                                                                  model.model.Y)[0, 0])

#################################################################

if criteria == 'log_pred_density_loo':

    #model.set_data(x, y)

    print("log_pred_density test objective")

    print(model.model.objective_function())
    print(-gp_experiments.gpy.libs.utils.metrics_computations.get_gaussian_log_lik(y.reshape(-1),
                                                                                  mu_loo_empirical.reshape(-1),
                                                                                  var_loo_empirical.reshape(-1)))

    ###############################################################

    print("log_pred_density test optimize")

    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    mu_loo_empirical, var_loo_empirical = get_loo(model, x, y)

    print(model.model.objective_function())
    print(-gp_experiments.gpy.libs.utils.metrics_computations.get_gaussian_log_lik(y.reshape(-1),
                                                                                  mu_loo_empirical.reshape(-1),
                                                                                  var_loo_empirical.reshape(-1)))

#################################################################

if criteria == 'crps_loo':

    #model.set_data(x, y)

    print("crps test objective")

    print(model.model.objective_function())
    print(-gp_experiments.gpy.libs.utils.metrics_computations.get_crps(mu_loo_empirical.reshape(-1),
                                                                       var_loo_empirical.reshape(-1),
                                                                       y.reshape(-1)))

    ###############################################################

    print("crps test optimize")

    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    mu_loo_empirical, var_loo_empirical = get_loo(model, x, y)

    print(model.model.objective_function())
    print(-gp_experiments.gpy.libs.utils.metrics_computations.get_crps(mu_loo_empirical.reshape(-1),
                                                                       var_loo_empirical.reshape(-1),
                                                                       y.reshape(-1)))

#################################################################

if criteria == 'standardized_mse_loo':

    #model.set_data(x, y)

    print("standardized_mse test objective")

    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_scaled_mse(mu_loo_empirical.reshape(-1),
                                                                       var_loo_empirical.reshape(-1),
                                                                       y.reshape(-1)))

    ###############################################################

    print("standardized_mse test optimize")

    a = model.model.optimize()
    print(model.model._objective_grads(model.model.optimizer_array))
    print(a.status)

    mu_loo_empirical, var_loo_empirical = get_loo(model, x, y)

    print(model.model.objective_function())
    print(gp_experiments.gpy.libs.utils.metrics_computations.get_scaled_mse(mu_loo_empirical.reshape(-1),
                                                                       var_loo_empirical.reshape(-1),
                                                                       y.reshape(-1)))

print(print(model.model.checkgrad(verbose=True, tolerance=10**(-8))))

# score = gp_experiments.gpy.libs.models.RationalMaternZeroMean.MinusCRPS()
#
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([-1]), var_loo=np.array([3**2])), 4))
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([-1]), var_loo=np.array([1])), 4))
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([-1]), var_loo=np.array([0.1 ** 2])), 4))
#
# print(round(score.evaluate(y=np.array([-1]),mu_loo=np.array([1]), var_loo=np.array([3 ** 2])), 4))
# print(round(score.evaluate(y=np.array([-1]),mu_loo=np.array([1]), var_loo=np.array([1])), 4))
# print(round(score.evaluate(y=np.array([-1]),mu_loo=np.array([1]), var_loo=np.array([0.1 ** 2])), 4))
#
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([1]), var_loo=np.array([3 ** 2])), 4))
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([1]), var_loo=np.array([1])), 4))
# print(round(score.evaluate(y=np.array([1]), mu_loo=np.array([1]), var_loo=np.array([0.1 ** 2])), 4))
