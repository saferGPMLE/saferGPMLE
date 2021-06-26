import numpy as np
import math
import libs.utils.metrics_computations
from scipy.stats import truncnorm
import paramz


def trainer(model, options, profiler=None, ipython_notebook=False):
    model, status = launch_sessions(
        model=model,
        ipython_notebook=ipython_notebook,
        profiler=profiler,
        **options
    )

    if profiler is not None:
        model = profiler(model)

    return model, status


def brutal_train(model, n, gtol=10**(-20), bfgs_factor=10, ipython_notebook=False,
                 log_bounds=[-3,5], profiler=None):

    log_rho_data = np.random.random((n, model.input_dim)) * (log_bounds[1] - log_bounds[0]) + log_bounds[0]

    rho_data = np.vectorize(lambda x: math.exp(x * math.log(10)))(log_rho_data)

    data = []

    for rho in rho_data:
        set_gpy_model_ls(model, rho)

        model, status = launch_sessions(
            model,
            optim_scheme=[[2, 2.0]],
            profiler=profiler,
            ipython_notebook=ipython_notebook,
            bfgs_factor=bfgs_factor,
            gtol=gtol
        )

        data_iter = {
                    'array': model.optimizer_array.copy(),
                    'cost': model.objective_function(),
                    'status': status
                    }

        data.append(data_iter)

    argmin_cost = np.array([x['cost'] for x in data]).argmin()

    model.optimizer_array = data[argmin_cost]['array']

    return model, None


def launch_sessions(
        model,
        optim_scheme,
        gtol,
        bfgs_factor,
        ipython_notebook,
        profiler
    ):

    status = 'untrained'

    for scheme in optim_scheme:
        model, status = custom_optimize_restarts(model=model, n_multistarts=scheme[0],
                                                 gtol=gtol, bfgs_factor=bfgs_factor,
                                                 std_perturbations=scheme[1],
                                                 profiler=profiler,
                                                 ipython_notebook=ipython_notebook)
        if profiler is not None:
            model = profiler(model)

    return model, status


def gaussian_random_init(model, num_restarts, std_perturbations):

    bounds = None
    current_point = model.optimizer_array.copy()

    draw = np.zeros([(num_restarts - 1), model.optimizer_array.shape[0]])

    if bounds is None:
        assert current_point.ndim == 1
        assert std_perturbations.ndim == 1

        for i in range(model.optimizer_array.shape[0]):
            draw[:, i] = np.random.normal(
                loc=current_point[i],
                scale=std_perturbations[i],
                size=num_restarts - 1
            )

    else:
        my_mean = current_point

        my_clip_a = np.array([x[0] for x in bounds])
        my_clip_b = np.array([x[1] for x in bounds])
        my_std = std_perturbations

        assert my_clip_a.shape == my_mean.shape
        assert my_clip_b.shape == my_mean.shape
        assert my_std.shape == my_mean.shape

        a, b = (my_clip_a - my_mean) / my_std, (my_clip_b - my_mean) / my_std

        assert a.shape == my_mean.shape
        assert b.shape == my_mean.shape
        assert a.ndim == 1

        for i in range(model.optimizer_array.shape[0]):
            draw[:, i] = truncnorm.rvs(
                a[i],
                b[i],
                loc=my_mean[i],
                scale=my_std[i],
                size=num_restarts - 1
            )

    return np.concatenate((current_point.reshape(1, -1), draw))


def custom_optimize_restarts(model, n_multistarts, gtol, bfgs_factor, std_perturbations, profiler, ipython_notebook):
    assert n_multistarts > 0, "multistarts should be > 0, {}".format(n_multistarts)

    std_perturbations_vector = std_perturbations * np.ones([model.optimizer_array.shape[0]])

    mean_index = [
        i for i in range(model.parameter_names_flat().shape[0])
        if 'constmap.C' in model.parameter_names_flat()[i]
    ]

    std_perturbations_vector[mean_index] = model.Y_normalized.std() * std_perturbations_vector[mean_index]

    inits = gaussian_random_init(model, n_multistarts, std_perturbations_vector)

    scores = []
    optimum = []
    statuses = []

    for x in inits:
        assert x.shape == model.optimizer_array.shape, "Shape issue."

        model.optimizer_array = x

        model, status = optimize_from_start(model, gtol, bfgs_factor, ipython_notebook)

        if profiler is not None:
            model = profiler(model)

        optimum.append(model.optimizer_array.copy())
        scores.append(model.objective_function())
        statuses.append(status)

    argmin = np.array(scores).argmin()

    model.optimizer_array = optimum[argmin]

    return model, statuses[argmin]


def analytical_mean_and_variance_optimization(model):

    estimated_mean = model.constmap.C.copy()[0]
    estimated_var = model.kern.variance.copy()[0]

    cost = model._objective_grads(model.optimizer_array)[0]

    try:
        analytical_mean, analytical_variance = get_beta_and_var_from_ls(model)
    except np.linalg.LinAlgError as e:
        print("Linalg error : {}, zero mean used instead, will be corrected if it lowers the NLL".format(e))
        set_gpy_model_var(model, estimated_var)
        model.constmap.C = estimated_mean

        return model

    set_gpy_model_var(model, analytical_variance[0,0])
    model.constmap.C = analytical_mean[0,0]

    analytical_parameter_cost = model._objective_grads(model.optimizer_array)[0]

    if analytical_parameter_cost >= cost:
        set_gpy_model_var(model, estimated_var)
        model.constmap.C = estimated_mean

    return model


def optimize_from_start(model, gtol, bfgs_factor, ipython_notebook, messages=False):
    if gtol is not None and bfgs_factor is not None:
        optim = model.optimize(messages=messages, max_iters=1000, start=None, clear_after_finish=False,
                        ipython_notebook=ipython_notebook, gtol=gtol, bfgs_factor=bfgs_factor)
    elif gtol is not None and bfgs_factor is None:
        optim = model.optimize(messages=messages, max_iters=1000, start=None, clear_after_finish=False,
                       ipython_notebook=ipython_notebook, gtol=gtol)
    elif gtol is None and bfgs_factor is not None:
        optim = model.optimize(messages=messages, max_iters=1000, start=None, clear_after_finish=False,
                       ipython_notebook=ipython_notebook, bfgs_factor=bfgs_factor)
    elif gtol is None and bfgs_factor is None:
        optim = model.optimize(messages=messages, max_iters=1000, start=None, clear_after_finish=False,
                       ipython_notebook=ipython_notebook)
    return model, optim.status


def get_beta_and_var_from_ls(model):
    K_inv = model.kern.variance.values.copy()[0] * model.posterior.woodbury_inv.copy()
    y = model.Y.values.copy()

    pred_matrix = np.ones([y.shape[0], 1])

    assert y.ndim == 2, "Ndim must be 2"
    assert pred_matrix.shape == y.shape, 'Shape issue.'
    assert K_inv.shape[0] == K_inv.shape[1] and K_inv.shape[0] == y.shape[0]

    beta = np.linalg.inv(pred_matrix.T @ K_inv @ pred_matrix) @ pred_matrix.T @ K_inv @ y

    assert y.shape == (pred_matrix @ beta).shape, 'Shape issue.'

    variance = (y - pred_matrix @ beta).T @ K_inv @ (y - pred_matrix @ beta) / model.X.shape[0]

    if variance <= 0:
        before_var = variance
        variance = np.ndarray([1,1])
        variance[0,0] = 10**(-10)
        print("Negative variance of {} brought back to {}.".format(before_var, variance))
    return beta, variance


def get_variance_estimate(mu_loo, var_loo, y, diagonal_var):
    assert isinstance(var_loo, np.ndarray) and isinstance(diagonal_var, float), 'Type issue'
    assert var_loo.ndim == 1, 'Shape issue'

    unit_var_scaled_mse = libs.utils.metrics_computations.get_scaled_mse(
        mu_loo, var_loo/diagonal_var, y)

    return unit_var_scaled_mse


def set_gpy_model_ls(gpy_model, value):
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError()
        value_copy = value.copy()
    elif isinstance(value, float):
        value_copy = np.asarray(value).reshape(-1)
    else:
        raise TypeError("{}".format(type(value)))

    gpy_model.kern.lengthscale = value_copy


def set_gpy_model_var(model, var):
    assert isinstance(var, float)
    assert not model.kern.variance.is_fixed

    model.kern.variance = var


def analytical_zero_mean_variance_optimization(model):

    estimated_var = model.kern.variance.copy()[0]

    cost = model._objective_grads(model.optimizer_array)[0]

    analytical_variance = get_zero_mean_and_var_from_ls(model)

    set_gpy_model_var(model, analytical_variance[0,0])

    analytical_parameter_cost = model._objective_grads(model.optimizer_array)[0]

    if analytical_parameter_cost >= cost:
        set_gpy_model_var(model, estimated_var)

    return model


def get_zero_mean_and_var_from_ls(model):
    K_inv = model.kern.variance.values.copy()[0] * model.posterior.woodbury_inv.copy()
    y = model.Y.values.copy()

    assert isinstance(y, np.ndarray), "Type issue"
    assert y.ndim == 2, "Ndim must be 2"
    assert y.shape[1] == 1, "Shape issue"

    assert isinstance(K_inv, np.ndarray), "Type issue"
    assert K_inv.ndim == 2, "Ndim must be 2"
    assert K_inv.shape[0] == K_inv.shape[1] and K_inv.shape[0] == y.shape[0], "Shape issue"

    variance = y.T @ K_inv @ y / \
               model.X.shape[0]

    if variance <= 0:
        before_var = variance
        variance = np.ndarray([1,1])
        variance[0,0] = 10**(-10)
        print("Negative variance of {} brought back to {}.".format(before_var, variance))
    return variance
