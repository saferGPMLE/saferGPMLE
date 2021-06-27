import time
from gp_experiments.gpy.libs.utils.gpy_estimation_lib import custom_optimize_restarts, gaussian_random_init, optimize_from_start
import numpy as np


def trainer_all(model, options, profiler=None, ipython_notebook=False, bench_type='single'):
    if bench_type == 'single':
        l = launch_sessions_all(model=model, ipython_notebook=ipython_notebook, profiler=profiler, **options)
    elif bench_type == 'monte-carlo':
        l = launch_sessions_all_monte_carlo(model=model, ipython_notebook=ipython_notebook, profiler=profiler, **options)
    else:
        raise ValueError(bench_type)
    return l


def launch_sessions_all(
        model,
        optim_scheme,
        gtol,
        bfgs_factor,
        ipython_notebook,
        profiler
):
    status = 'untrained'
    l = {}
    idx = 0
    start = time.time()
    for scheme in optim_scheme:
        model, status = custom_optimize_restarts(model=model, n_multistarts=scheme[0],
                                                 gtol=gtol, bfgs_factor=bfgs_factor,
                                                 std_perturbations=scheme[1],
                                                 profiler=profiler,
                                                 ipython_notebook=ipython_notebook)
        if profiler is not None:
            model = profiler(model)
        idx += 1
        end = time.time()
        # For studying the improvements over the restarts
        # print("\nscheme : {}, cost : {}".format(scheme, model.objective_function()))
        l[str(scheme) + "_nll_" + str(idx)] = model.objective_function()
        l[str(scheme) + "_time_" + str(idx)] = end - start

    return l


def launch_sessions_all_monte_carlo(
        model,
        optim_scheme,
        gtol,
        bfgs_factor,
        ipython_notebook,
        profiler
    ):

    status = 'untrained'
    for scheme in optim_scheme:
        model, l = custom_optimize_restarts_misc(model=model, n_multistarts=scheme[0],
                                                 gtol=gtol, bfgs_factor=bfgs_factor,
                                                 std_perturbations=scheme[1],
                                                 profiler=profiler,
                                                 ipython_notebook=ipython_notebook)
        if profiler is not None:
            model = profiler(model)

    return l


def custom_optimize_restarts_misc(model, n_multistarts, gtol, bfgs_factor, std_perturbations, profiler, ipython_notebook):
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
    idx = 0
    l = {}

    for x in inits:
        assert x.shape == model.optimizer_array.shape, "Shape issue."

        model.optimizer_array = x

        start = time.time()

        model, status = optimize_from_start(model, gtol, bfgs_factor, ipython_notebook)

        if profiler is not None:
            model = profiler(model)

        end = time.time()

        optimum.append(model.optimizer_array.copy())
        scores.append(model.objective_function())
        statuses.append(status)

        idx += 1
        l["nll_" + str(idx)] = model.objective_function()
        l["time_" + str(idx)] = end - start

    argmin = np.array(scores).argmin()

    model.optimizer_array = optimum[argmin]

    return model, l
