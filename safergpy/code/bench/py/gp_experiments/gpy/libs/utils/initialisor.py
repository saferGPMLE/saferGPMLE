import pandas as pd
import numpy as np
import math
import libs.utils.metrics_computations
from scipy.stats import truncnorm
import paramz
from libs.utils.gpy_estimation_lib import set_gpy_model_ls, optimize_from_start, set_gpy_model_var

def std_init(model, use_dimension, profiler, fix_var, is_zero_mean):
    assert not (fix_var and profiler is not None)

    x = model.X
    y = model.Y

    if not is_zero_mean:
        model.constmap.C = y.mean()

    if not fix_var:
        set_gpy_model_var(model, y.var())

    if use_dimension:
        untrain_ls = math.sqrt(x.shape[1]) * x.std(0)
    else:
        untrain_ls = x.std(0)

    set_gpy_model_ls(model, untrain_ls)

    if profiler is not None:
        model = profiler(model)

    return model

def custom_init(model):
    return model

def grid_init(model, isotropic, profiler, fix_var, is_zero_mean):
    assert not (fix_var and profiler is not None)

    if isotropic:
        raise ValueError("No sense if inputs can be standardized")
    else:
        normalized_diameter = math.sqrt(model.X.shape[1])
        nominal_scales = normalized_diameter * (model.X.max(0) - model.X.min(0))

    grid_factor = np.logspace(math.log10(1 / 50), math.log10(2), num=5, base=10.0)

    scores = []

    for fact in grid_factor:
        set_gpy_model_ls(model, fact * nominal_scales)

        if profiler is not None:
            model = profiler(model)

        scores.append(model._objective_grads(model.optimizer_array)[0])

    #print("grid : {}".format(grid))
    #print("scores : {}".format(scores))

    best_model_index = np.argmin(scores)

    set_gpy_model_ls(model, grid_factor[best_model_index] * nominal_scales)

    ###############################

    if profiler is not None:
        model = profiler(model)

    return model
