import numpy as np
import libs.utils.gpy_estimation_lib as gpy_estimation_lib
# -*- coding: utf-8 -*-

# def _eval_obj_with_delta (model, idx_param, delta):
#
#     # We don't want to modify the original model
#     M = model.copy ()
#
#     param = M.optimizer_array
#     # note: model.optimizer_array returns a value, not a reference
#     #        -> no need to call .copy()
#
#     param[idx_param] += delta
#     return M._objective_grads (param)
#
#
# def centraldiff_oneparam (model, idx_param, delta):
#
#     obj_p, grad_p = _eval_obj_with_delta (model, idx_param, +delta)
#     obj_m, grad_m = _eval_obj_with_delta (model, idx_param, -delta)
#
#     return (obj_p - obj_m) / (2 * delta), (grad_p - grad_m) / (2 * delta)
#
#
# def do_diagnose_partialdiff (model, idx_param):
#
#     # Value of the parameter
#     t = model.optimizer_array[idx_param]
#
#     _, grad = _eval_obj_with_delta (model, idx_param, 0.0)
#     print ("partialdiff GPy -->  %+.5e" % (grad[idx_param]))
#
#     delta_max = abs(t) * 1e-2
#     for k in range(14):
#         delta = delta_max * 10**(-k)
#         diff, _ = centraldiff_oneparam (model, idx_param, delta)
#         print ("delta=%8.2e  -->  %+.5e" % (delta, diff))

def get_cost_and_grad_and_hessian(model, lengthscale_value, mean_value, variance_value):
    model.Mat52.lengthscale = lengthscale_value

    beta_estimate, variance_estimate = gpy_estimation_lib.get_beta_and_var_from_ls(model)
    if mean_value is None:
        mean_value = beta_estimate
    if variance_value is None:
        variance_value = variance_estimate

    model.constmap.C = mean_value[0, 0]
    model.Mat52.variance = variance_value[0, 0]

    obj, grad, model = get_cost_and_grad(model, lengthscale_value, mean_value, variance_value)
    hessian, model = get_hessian(model)

    return obj, grad, hessian, model


def get_cost_and_grad(model, lengthscale_value, mean_value, variance_value):
    obj_grad = model._objective_grads(model.optimizer_array)

    return obj_grad[0], obj_grad[1], model


def get_hessian(model):
    # log_exp = paramz.transformations.Logexp()

    eps = 0.001
    n_var = model.optimizer_array.shape[0]

    array = model.optimizer_array.copy()

    hessian = np.zeros([n_var, 0])

    model.optimizer_array = array

    for i in range(n_var):
        d = np.zeros([n_var])
        d[i] = eps

        _, grad_minus_half_eps = model._objective_grads(model.optimizer_array - d)
        model.optimizer_array = array
        _, grad_plus_half_eps = model._objective_grads(model.optimizer_array + d)
        model.optimizer_array = array

        hessian_row = (grad_plus_half_eps - grad_minus_half_eps) / (2 * eps)

        hessian = np.concatenate((hessian, hessian_row.reshape([n_var, 1])), axis=1)

    return hessian, model
