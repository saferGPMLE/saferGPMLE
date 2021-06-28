import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import libs.utils.gpy_finite_diff as gpy_finite_diff
import libs.utils.gpy_estimation_lib as gpy_estimation_lib

from GPy.util.linalg import pdinv, dpotrs
from GPy.util import diag
import sklearn.linear_model


def plot_paramz_likelihood_path(model, label):

    nominal = np.array([27.04301504, 83.37540132])
    param_one = nominal.copy()
    param_two = model.X.std(0)

    scipy_param_one = model.kern.lengthscale.constraints.properties()[0].finv(param_one)
    scipy_param_two = model.kern.lengthscale.constraints.properties()[0].finv(param_two)

    grid_1d = np.linspace(-0.3, 1.5, 1000)

    y_1d = []

    for i in range(grid_1d.shape[0]):
        x = grid_1d[i]
        scipy_param = x * scipy_param_one + (1 - x) * scipy_param_two

        model.kern.lengthscale.optimizer_array = scipy_param.copy()
        model = gpy_estimation_lib.analytical_mean_and_variance_optimization(model)

        y_1d.append(model.objective_function())

    plt.plot(grid_1d, y_1d, label=label)


def plot_likelihood_path(model, estimate_mean=True, estimate_var=True):
    param_one = np.array([27.04301504, 83.37540132])
    param_two = np.array([8.76182561, 21.70946319])

    mean_one = 1210.116506

    variance_one = 2274398.204448

    mean_two = 176.754115

    variance_two = 18221.51397

    grid_1d = np.linspace(0, 1, 1000)

    y_1d = []
    gradient_norm = []

    for i in range(grid_1d.shape[0]):
        x = grid_1d[i]
        param = x * param_one + (1 - x) * param_two

        if estimate_mean:
            mean_value = None
        else:
            mean_value = np.array([[x * mean_one + (1 - x) * mean_two]])

        if estimate_var:
            variance_value = None
        else:
            variance_value = np.array([[x * variance_one + (1 - x) * variance_two]])

        obj, grad, hessian, model = gpy_finite_diff.get_cost_and_grad_and_hessian(model, param, mean_value, variance_value)

        if i == 0:
            print("first mode, obj : {}, grad : {}, hessian : \n {}, \n spec hessian : {}".format(obj, grad,
                                                                                                  hessian,
                                                                                                  np.linalg.eig(
                                                                                                      hessian)[0]))
        elif i == grid_1d.shape[0] - 1:
            print("second mode, obj : {}, grad : {}, hessian : \n {}, \n spec hessian : {}".format(obj, grad,
                                                                                                   hessian,
                                                                                                   np.linalg.eig(
                                                                                                       hessian)[0]))

        y_1d.append(obj)
        gradient_norm.append((grad ** 2).sum())

    plot_xaxis = "path : lengthscales"
    if not estimate_mean:
        plot_xaxis = plot_xaxis + ', mean'

    if not estimate_var:
        plot_xaxis = plot_xaxis + ', var'

    plt.figure()
    plt.plot(grid_1d, y_1d)
    plt.title("NLL vs Path")
    plt.xlabel(plot_xaxis)
    plt.ylabel("Negative log likelihood")
    plt.show()

    plt.figure()
    plt.semilogy()
    plt.plot(grid_1d, gradient_norm)
    plt.title("Log gradient norm vs Path")
    plt.xlabel(plot_xaxis)
    plt.ylabel("Log gradient norm of negative log likelihood")
    plt.show()


def plot_neg_likelihood_var(model):

    var_init = model.Mat52.variance.values[0]
    cost_var_init = model._objective_grads(model.optimizer_array)[0]

    grid_1d = np.linspace(-1, 1, 2001)

    grid_1d = [var_init * math.exp(x * math.log(10)) for x in grid_1d]

    y_1d = []
    for x in grid_1d:
        model.Mat52.variance = x
        y_1d.append((model._objective_grads(model.optimizer_array)[0]))

    plt.figure()

    plt.semilogx()
    plt.plot(grid_1d, y_1d)
    plt.title("Negative log likelihood vs var : lengthscales : [{}, {}]".format(model.Mat52.lengthscale[0],
                                                                                model.Mat52.lengthscale[1]))
    plt.xlabel("var")
    plt.ylabel("Negative log likelihood")
    plt.vlines(var_init, ymin=min(y_1d), ymax=max(y_1d), label='estimated_var : {0:.3f}, nll : {1:.3f}'.format(var_init, cost_var_init))
    plt.legend()
    plt.show()


def plot_multistart_optimization(model, n, mean_value,
                                 variance_value,
                                 optimum,
                                 init_type):

    model.constmap.C = mean_value
    model.Mat52.variance = variance_value

    bounds = [-1, 1]

    log_rho_data = np.random.random((n, 2)) * (bounds[1] - bounds[0]) + bounds[0] + np.log10(optimum)

    rho_data = np.exp(log_rho_data * math.log(10))

    data = pd.DataFrame({'rho1': [], 'rho2': [], 'sigma2': [], 'm': [], 'cost': [], 'status': []})

    for rho in rho_data:
        model.Mat52.lengthscale = rho

        if init_type == 'profiled':
            model = gpy_estimation_lib.analytical_mean_and_variance_optimization(model)
        elif init_type == 'classic':
            model.constmap.C = model.Y.mean()
            model.kern.variance = model.Y.var()
        else:
            ValueError(init_type)

        optim = model.optimize()
        data = data.append(pd.DataFrame({'rho1': [model.Mat52.lengthscale[0]],
                                         'rho2': [model.Mat52.lengthscale[1]],
                                         'sigma2': model.Mat52.variance,
                                         'm': [model.constmap.C],
                                         'cost': [model._objective_grads(model.optimizer_array)[0]],
                                         'status': optim.status}),
                           ignore_index=True)

    colors = {"Errorb'ABNORMAL_TERMINATION_IN_LNSRCH'": 'red', 'Converged': 'blue', 'Maximum number of f evaluations reached': 'green'}

    if not data['status'].apply(lambda x: x in colors.keys()).min():
        raise ValueError('Unknown status : {}'.format(data['status'].unique()))

    plt.figure()

    plt.scatter(x=np.log10(data['rho1']), y=np.log10(data['rho2']),
                c=data['status'].apply(lambda x: colors[x]))
    plt.scatter(x=math.log10(optimum[0]), y=math.log10(optimum[1]), c='k')

    plt.xlabel("ln(rho_1)")
    plt.ylabel("ln(rho_2)")

    plt.vlines(x=math.log(10) * bounds[0] + math.log10(optimum[0]),
               ymin=math.log(10) * bounds[0] + math.log10(optimum[1]),
               ymax=math.log(10) * bounds[1] + math.log10(optimum[1]),
               linestyles="--", colors="g")
    plt.vlines(x=math.log(10) * bounds[1] + math.log10(optimum[0]),
               ymin=math.log(10) * bounds[0] + math.log10(optimum[1]),
               ymax=math.log(10) * bounds[1] + math.log10(optimum[1]),
               linestyles="--", colors="g")

    plt.hlines(y=math.log(10) * bounds[0] + math.log10(optimum[1]),
               xmin=math.log(10) * bounds[0] + math.log10(optimum[0]),
               xmax=math.log(10) * bounds[1] + math.log10(optimum[0]),
               linestyles="--", colors="g")
    plt.hlines(y=math.log(10) * bounds[1] + math.log10(optimum[1]),
               xmin=math.log(10) * bounds[0] + math.log10(optimum[0]),
               xmax=math.log(10) * bounds[1] + math.log10(optimum[0]),
               linestyles="--", colors="g")

    plt.plot([math.log10(optimum[0]) - 2, math.log10(optimum[0]) + 2],
             [math.log10(optimum[1]) - 2, math.log10(optimum[1]) + 2],
             label='constant anisotropy')

    plt.legend()

    plt.title(init_type)

    plt.show()

    #############################################

    plt.figure()

    plt.scatter(x=np.log10(data['rho1']), y=np.log10(data['sigma2']),
                c=data['status'].apply(lambda x: colors[x]))
    plt.scatter(x=math.log10(optimum[0]), y=math.log10(variance_value), c='k')

    plt.vlines(x=math.log(10) * bounds[0] + math.log10(optimum[0]), ymin=np.log10(data['sigma2']).min(), ymax=np.log10(data['sigma2']).max(),
               linestyles="--", colors="g")
    plt.vlines(x=math.log(10) * bounds[1] + math.log10(optimum[0]), ymin=np.log10(data['sigma2']).min(), ymax=np.log10(data['sigma2']).max(),
               linestyles="--", colors="g")

    plt.plot([np.log10(data['rho1']).min(), np.log10(data['rho1']).max()],
             [math.log10(variance_value) - (math.log10(optimum[0]) - np.log10(data['rho1']).min())*5,
              math.log10(variance_value) + (np.log10(data['rho1']).max() - math.log10(optimum[0]))*5], label='constant microergodicity')

    plt.xlabel("ln(rho_1)")
    plt.ylabel("ln(sigma2)")

    plt.legend()

    plt.title(init_type)

    plt.show()

    return data


def get_noise_level(x, y):
    sk_model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    X_data = np.concatenate((np.array(x).reshape(-1, 1), (np.array(x)**2).reshape(-1, 1)), axis=1)
    Y_data = np.array(y).reshape(-1, 1)

    sk_model.fit(X_data, Y_data)

    print("noise level (std) : {}".format((Y_data - sk_model.predict(X_data)).std(ddof=3)))


def plot_taylor(model, idx_param, diagonalize=False, width=1e-2, n=1000):
    obj_value, grad = model._objective_grads(model.optimizer_array)

    print("obj value : {}".format(obj_value))

    hessian, model = gpy_finite_diff.get_hessian(model)

    if diagonalize:
        v, W = np.linalg.eig(hessian)
        direction = W[:, idx_param]
    else:
        direction = np.zeros([model.optimizer_array.shape[0]])
        direction[idx_param] = 1

    array = model.optimizer_array.copy()

    dx_vector = np.linspace(-width / 2, width / 2, n)

    y = []
    y_order_1 = []
    y_order_2 = []

    for dx in dx_vector:
        d = direction.copy() * dx
        obj, _ = model._objective_grads(model.optimizer_array + d)
        y.append(obj)
        model.optimizer_array = array.copy()
        y_order_1.append(obj_value + (d * grad).sum())
        y_order_2.append(obj_value + (d * grad).sum() + 0.5 * (d.reshape(1, -1) @ hessian @ d.reshape(-1, 1))[0, 0])

    fig, ax = plt.subplots()

    plt.plot(dx_vector, y, label="NLL")

    ##############################################

    sk_model = sklearn.linear_model.LinearRegression(fit_intercept=True)

    X_data = np.concatenate((np.array(dx_vector).reshape(-1, 1), (np.array(dx_vector)**2).reshape(-1, 1)), axis=1)
    Y_data = np.array(y).reshape(-1, 1)

    sk_model.fit(X_data, Y_data)

    plt.plot(dx_vector, sk_model.predict(X_data), label='Best linear fit')

    ##############################################

    ax.ticklabel_format(useOffset=False)
    plt.axvline(x=0, color='red', label='')
    plt.legend()
    plt.show()

    get_noise_level(dx_vector, y)


def decompose_all(model, idx_param, diagonalize=False, width=1e-2, n=1000):
    obj_value, grad = model._objective_grads(model.optimizer_array)

    print("obj value : {}".format(obj_value))

    hessian, model = gpy_finite_diff.get_hessian(model)

    if diagonalize:
        v, W = np.linalg.eig(hessian)
        # Slow variation direction : array([ 9.99997623e-01, -2.06640309e-03, -4.50014843e-04, -5.31242312e-04])
        direction = W[:, idx_param]
        eig_value = v[idx_param]
    else:
        direction = np.zeros([model.optimizer_array.shape[0]])
        direction[idx_param] = 1

    array = model.optimizer_array.copy()

    dx_vector = np.linspace(-width / 2, width / 2, n)

    y_data = []
    y_reg = []

    for dx in dx_vector:
        d = direction.copy() * dx
        model.optimizer_array = model.optimizer_array + d

        m = model.mean_function.f(model.X)

        variance = model.likelihood.gaussian_variance(model.Y_metadata)

        YYT_factor = model.Y_normalized - m

        K = model.kern.K(model.X)

        Ky = K.copy()
        diag.add(Ky, variance)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)

        y_reg.append(0.5*(- model.Y.shape[1] * W_logdet))
        y_data.append(0.5*(- np.sum(alpha * YYT_factor)))

        model.optimizer_array = array.copy()

    plt.subplots(1, 2)
    if diagonalize:
        plt.suptitle("Eigen axis : {}, eigen value : {} \n eigen vector: ({})".format(idx_param,
                                                                                      "{:.4E}".format(eig_value),
                                                                                      ','.join(["{:.6}".format(x) for x in direction])))
    else:
        plt.suptitle("Axis".format(idx_param))

    plt.subplot(1, 2, 1)
    plt.title("Data term")
    plt.plot(dx_vector, y_data, label="Data term")

    plt.subplot(1, 2, 2)
    plt.title("Regularization term")
    plt.plot(dx_vector, y_reg, label="Regularization term")

    plt.show()

    print("Regularizer")
    get_noise_level(dx_vector, y_reg)
    print("Data")
    get_noise_level(dx_vector, y_data)


def plot_model(model, x_train, y_train, x_grid, y_grid, title):
    _, ax = plt.subplots()

    plt.plot(x_grid, y_grid, 'k', label='truth')
    plt.plot(x_train, y_train, 'ko', label='observed')

    mu, var = model.predict(x_grid)

    var[var <= 0] = 0

    plt.plot(x_grid, mu, 'r', label='mu')
    plt.plot(x_grid, mu - np.vectorize(math.sqrt)(var), 'b', label='mu - sigma')
    plt.plot(x_grid, mu + np.vectorize(math.sqrt)(var), 'b', label='mu + sigma')

    ax.fill_between(x_grid.reshape(-1), (mu - np.vectorize(math.sqrt)(var)).reshape(-1),
                    (mu + np.vectorize(math.sqrt)(var)).reshape(-1), color='#539caf', alpha=0.4)

    plt.legend()
    plt.title(title)
    plt.show()
