import pandas as pd
from libs.models.CustomGPy import CustomGPy
import numpy as np
import math
from libs.utils.gpy_estimation_lib import analytical_mean_and_variance_optimization
import matplotlib.pyplot as plt
import GPy

data = pd.read_csv('/Users/sebastien/git-repos/test-functions/data/doe/g10_20d.csv', sep = ',', index_col = 0)

predictors = [a for a in data.columns if 'x' in a]
outputs = ['f_1']

x_train = data[predictors].values
z_train = data[outputs].values

#x_train = np.linspace(0, 10, 5).reshape(-1,1)
#z_train = np.sin(x_train)

#x_test = np.linspace(-50, 60, 1001).reshape(-1,1)


mean_function = GPy.mappings.constant.Constant(input_dim=x_train.shape[1], output_dim=1, value=0.0)

kernel_function_one = GPy.kern.Matern52(input_dim=x_train.shape[1],
                                variance=1.0, lengthscale=1.0, ARD=True)

model_one = GPy.models.GPRegression(x_train, z_train, mean_function=mean_function,
                                     Y_metadata=None, normalizer=None,
                                     noise_var=0, kernel=kernel_function_one)

model_one.Gaussian_noise.variance.fix()

kernel_function_two = GPy.kern.RationalMatern(p = 2, input_dim=x_train.shape[1],
                                variance=1.0, lengthscale=1.0, ARD=True)

model_two = GPy.models.GPRegression(x_train, z_train, kernel=kernel_function_two,
                                     Y_metadata=None, normalizer=None,
                                     noise_var=0, mean_function=mean_function)

model_two.Gaussian_noise.variance.fix()

model_one.kern.lengthscale = 1000
model_two.kern.lengthscale = 1000

print(model_one.objective_function())
print("")
print(model_two.objective_function())
print("")
print("")
print(model_one.kern.K(x_train[:3, :], x_train[:3, :]))
print("")
print(model_two.kern.K(x_train[:3, :], x_train[:3, :]))

print(((model_one.kern.K(x_train[:3, :], x_train[:3, :]) - model_two.kern.K(x_train[:3, :], x_train[:3, :]))**2).sum())
print((model_one.kern.K(x_train[:3, :], x_train[:3, :]) == model_two.kern.K(x_train[:3, :], x_train[:3, :])).all())
print(model_one.objective_function() == model_two.objective_function())

###############################################################################################


model = CustomGPy(init="cheated_classic",
                  stopping_criterion="strict",
                  do_profiling=True,
                  do_restarts=True,
                  n_multistarts=2,
                  n_iterations=3)

model.set_data(x_train, z_train)

###############################################################################################

plt.figure()

plt.plot(np.linspace(-3, 3, 1000), model.model.kern.K(np.linspace(-3, 3, 1000).reshape(-1, 1), np.array([[0]])))

plt.xlabel('h')
plt.ylabel('k(h)')

plt.title("Matern 5/2")

plt.show()

###############################################################################################

model.train()

def plot_model(x, model, x_test, x_train, z_train):
    model.model.kern.lengthscale = x

    plt.plot(x_train, z_train, 'bo', label = 'training')
    plt.plot(x_test, model.predict(x_test)[0], label = 'prediction')

    alpha = np.linalg.inv(model.model.kern.K(x_train)) @ z_train

    basis_values = np.tile(alpha.reshape(1, -1), reps = [x_test.shape[0], 1]) * model.model.kern.K(x_test, x_train)

    #for i in range(model.model.Y.shape[0]):
    #    plt.plot(x_test, basis_values[:, i], label = str(i))

    plt.plot(x_test, basis_values.sum(1), label = 'sum basis')

    plt.legend()
    plt.show()

def plot_sample(x, model, x_test):
    model.model.kern.lengthscale = x
    for i in range(10):
        path = np.random.multivariate_normal(mean = np.zeros(x_test.shape[0]), cov = model.model.kern.K(x_test))
        plt.plot(x_test, path, label = str(i))

    plt.legend()
    plt.show()




ls = model.model.Mat52.lengthscale.copy()

log_scale_multi = np.linspace(-1, 2, 1000)

cost = []

for x in log_scale_multi:
    model.model.Mat52.lengthscale = ls*math.exp(x * math.log(10))
    #model.model = analytical_mean_and_variance_optimization(model.model)
    cost.append(model.get_objective_value())

plt.plot(log_scale_multi[:len(cost)], cost)
plt.xlabel("log10 scale delta")
plt.ylabel("ALL")
plt.title("Evolution of ALL with uniform scaling of the scales.")
