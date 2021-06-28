import sys, os

import pandas as pd
import scipy.stats
import numpy as np
import math
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

# Problably possible to be cleaner with a proper package.
if "__file__" in globals():
    sys.path.append(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..'))

import libs.utils.metrics_computations

#######################################################


def mc_validation(x_train, y_train, ModelClass, n_sample, x_test=None, y_test=None):
    model = ModelClass(x_train, y_train)
    model.train()

    y_test_np = y_test.values.reshape(-1, 1)

    ########################################################

    if x_test is None and y_test is None:
        model.model.set_loo_posterior_metrics()
        post_mean = model.model.loo_mean
        post_var = model.model.loo_var
    else:
        post_mean, post_var = model.predict(x_test)

    errors = y_test_np - post_mean

    trained_model_mean_error = errors.mean()
    trained_model_var_error = errors.var()

    normalized_error = errors / np.sqrt(post_var)

    trained_model_mean_normalized_error = normalized_error.mean()
    trained_model_var_normalized_error = normalized_error.var()

    ########################################################

    lengthscales = model.model.model.kern.lengthscale.values.copy()
    c_var = model.model.model.kern.variance.values.copy()[0]
    c_mean = model.model.model.constmap.C.values.copy()[0]

    var_range_log10 = [-15, 15]
    scale_range_log10 = [-5, 15]

    unit_uniform_draw = np.random.random((n_sample, lengthscales.shape[0] + 2))

    sample_means = np.exp(math.log(2) * (2 * unit_uniform_draw[:, 0] - 1) + math.log(c_mean))

    sample_vars = np.exp(math.log(10) * ((unit_uniform_draw[:, 1] * (var_range_log10[1] - var_range_log10[0]) + var_range_log10[0]) + math.log10(c_var)))

    sample_lengthscales = np.exp(math.log(10) * (np.tile(np.log10(lengthscales), reps=(n_sample, 1)) +
                                                 (unit_uniform_draw[:, 2:] * (scale_range_log10[1] - scale_range_log10[0]) + scale_range_log10[0])))

    mean_errors = []
    var_errors = []

    mean_normalized_error = []
    var_normalized_error = []

    if x_test is None and y_test is None:
        y_test = y_train

    for i in range(n_sample):
        model.model.model.kern.lengthscale = sample_lengthscales[i, :]
        model.model.model.kern.variance = sample_vars[i]
        model.model.model.constmap.C = sample_means[i]

        if x_test is None and y_test is None:
            model.model.set_loo_posterior_metrics()
            post_mean = model.model.loo_mean
            post_var = model.model.loo_var
        else:
            post_mean, post_var = model.predict(x_test)

        assert post_mean.shape == y_test_np.shape and post_var.shape == y_test_np.shape, "Shape issue"

        errors = y_test_np - post_mean

        mean_errors.append(errors.mean())
        var_errors.append(errors.var())

        normalized_error = errors/np.sqrt(post_var)

        mean_normalized_error.append(normalized_error.mean())
        var_normalized_error.append(normalized_error.var())

    plot_mse_decomposition(mean_errors, var_errors, y_test_np.std(), trained_model_mean_error, trained_model_var_error)

    plot_mse_decomposition(mean_normalized_error, var_normalized_error, 1, trained_model_mean_normalized_error, trained_model_var_normalized_error)

    return mean_errors, var_errors, mean_normalized_error, var_normalized_error, sample_lengthscales, sample_vars, sample_means


def plot_mse_decomposition(mean_errors, var_errors, empirical_std, trained_model_mean_error, trained_model_var_error):

    std_errors = [math.sqrt(x) for x in var_errors]
    abs_mean_errors = [abs(x) for x in mean_errors]

    lower = min(-max(abs_mean_errors), -max(std_errors), -2*empirical_std)
    upper = max(max(abs_mean_errors), max(std_errors), 2*empirical_std)

    plt.figure()

    plt.plot(std_errors, mean_errors, 'o', label="Sampled GP")
    plt.vlines(x=empirical_std, ymin=lower, ymax=upper, label='Observations std')

    circle1 = plt.Circle((0, 0), 0.5*empirical_std, color='k', fill=False)
    circle2 = plt.Circle((0, 0), empirical_std, color='k', fill=False)
    circle3 = plt.Circle((0, 0), 2*empirical_std, color='k', fill=False)

    plt.scatter([math.sqrt(trained_model_var_error)], [trained_model_mean_error], label="trained model", color='k')

    ax = plt.gca()

    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.add_artist(circle3)

    plt.xlabel("std err")
    plt.ylabel("mean err")

    plt.xlim([lower, upper])
    plt.xlim([lower, upper])

    plt.title("MSE decomposition")
    plt.legend()

    plt.show()


def get_gaussian_log_lik(y_test, post_mean, post_var):
    return libs.utils.metrics_computations(y_test, post_mean, post_var)


def get_chi2_alpha_coverage(y_test, y_pred, y_var, alpha):
    assert y_pred.shape == y_var.shape and y_pred.shape == y_test.shape, "Shape issue"
    assert isinstance(alpha, float), "alpha must be float"

    assert np.all(y_var > 0)

    standardized_residuals = (y_test - y_pred)/np.sqrt(y_var)

    lower = scipy.stats.chi2.ppf((1 - alpha) / 2, df=1)
    upper = scipy.stats.chi2.ppf(1 - (1 - alpha) / 2, df=1)

    is_alpha_credible = np.logical_and(lower <= standardized_residuals, standardized_residuals <= upper).astype(float)

    return is_alpha_credible.mean()


def update_with_error_statistics(post_mean, post_var, truth, metrics):
    assert post_mean.shape == post_var.shape and post_mean.shape == truth.shape, "Shape issue."

    assert np.all(post_var > 0)

    errors = truth - post_mean
    standard_errors = (truth - post_mean)/np.sqrt(post_var)

    metrics['error_mean'].append(errors.mean())
    metrics['error_var'].append(errors.var())
    metrics['error_skew'].append(scipy.stats.skew(errors.reshape(-1)))
    metrics['error_kurt'].append(scipy.stats.kurtosis(errors.reshape(-1)))
    metrics['standard_error_mean'].append(standard_errors.mean())
    metrics['standard_error_var'].append(standard_errors.var())
    metrics['standard_error_skew'].append(scipy.stats.skew(standard_errors.reshape(-1)))
    metrics['standard_error_kurt'].append(scipy.stats.kurtosis(standard_errors.reshape(-1)))

    return metrics


def get_mse(y_test, post_mean):
    assert y_test.shape == post_mean.shape, "Shape issue : {} and {}".format(y_test, post_mean)
    return ((y_test - post_mean) ** 2).mean()


def get_mae(y_test, post_mean):
    assert y_test.shape == post_mean.shape, "Shape issue : {} and {}".format(y_test.shape, post_mean.shape)
    return (abs(y_test - post_mean)).mean()


def get_gaussian_alpha_coverage(y_test, y_pred, y_var, alpha):
    return libs.utils.metrics_computations.get_gaussian_alpha_coverage(y_test, y_pred, y_var, alpha)

# def get_classifier_logloss(y_test, x_test, model):
#     prob_feasibility = model.get_cdf(0.0, x_test)
#
#     assert prob_feasibility.shape == y_test.shape, "Shape issue : {} and {}".format(prob_feasibility.shape,
#                                                                                     y_test.shape)
#
#     feasible_y_test = (y_test <= 0).astype(int)
#
#     non_zero_loss_key = (prob_feasibility != feasible_y_test)
#
#     if not non_zero_loss_key.max():
#         return 0
#
#     feasible_y_test = feasible_y_test[non_zero_loss_key]
#     prob_feasibility = prob_feasibility[non_zero_loss_key]
#
#     assert prob_feasibility.shape == feasible_y_test.shape, \
#         "Shape issue : {} and {}".format(prob_feasibility.shape, feasible_y_test.shape)
#
#     losses = -(feasible_y_test * np.log(prob_feasibility) + (1 - feasible_y_test) * np.log(1 - prob_feasibility))
#
#     return losses[(prob_feasibility != feasible_y_test)].mean() * non_zero_loss_key.mean()


def get_gaussian_classifier_mse(y_test, y_pred, y_var):
    assert y_pred.shape == y_var.shape, "Shape issue"

    assert np.all(y_var > 0)

    y_sd = np.vectorize(math.sqrt)(y_var)

    prob_feasibility = scipy.stats.norm.cdf((0.0 - y_pred)/y_sd)

    feasible_y_test = (y_test <= 0).astype(int)

    assert prob_feasibility.shape == feasible_y_test.shape, "Shape issue : {} and {}".format(
                                                        prob_feasibility.shape,
                                                        feasible_y_test.shape)
    return ((feasible_y_test - prob_feasibility)**2).mean()


def get_improvement_mse(y_test, previous_min, ei):
    assert ei.shape == y_test.shape and previous_min.shape == y_test.shape, "Shape issue."

    return (np.where(y_test >= previous_min, -ei, (previous_min - y_test) - ei)**2).mean()


def get_objective_improvement_gaussian_likelihood(y_test, previous_min, post_mean, post_var):
    assert y_test.shape == previous_min.shape, "Shape issue : {} and {}".format(y_test.shape, previous_min.shape)
    assert y_test.shape == post_mean.shape, "Shape issue : {} and {}".format(y_test.shape, post_mean.shape)
    assert y_test.shape == post_var.shape, "Shape issue : {} and {}".format(y_test.shape, post_var.shape)

    assert np.all(post_var > 0)

    zero_improvement_density = 1 - scipy.stats.norm.cdf((previous_min - post_mean)/np.sqrt(post_var))

    # This may throw a "RuntimeWarning:divide by zero encountered in log" warning.
    # This value maybe not be selected in the np.where. Not that postponing this in np.where wouldn't
    # supress the warning.
    zero_improvement_log_density = np.log(zero_improvement_density)

    non_zero_improvement_log_density = scipy.stats.norm.logpdf(x=previous_min - y_test,
                                                               loc=post_mean, scale=np.sqrt(post_var))

    assert y_test.shape == zero_improvement_log_density.shape, \
        "Shape issue : {} and {}".format(zero_improvement_log_density.shape, y_test.shape)
    assert y_test.shape == non_zero_improvement_log_density.shape, \
        "Shape issue : {} and {}".format(non_zero_improvement_log_density.shape, y_test.shape)

    return np.where(y_test >= previous_min, zero_improvement_log_density, non_zero_improvement_log_density).mean()


def get_ks_gaussian_comparison_statistic(y_test, post_mean, post_var):
    return libs.utils.metrics_computations.get_ks_gaussian_comparison_statistic(y_test, post_mean, post_var)

#######################################################


def update_metrics_with_posterior(y_test, post_mean, post_var,
                                  metrics, alpha, output, row,
                                  c_mean, ls, c_var, cost, status):

    assert y_test.shape == post_mean.shape and y_test.shape == post_var.shape, "Shape issue."

    standard_y_test = (y_test - post_mean) / np.vectorize(math.sqrt)(post_var)

    mse = (y_test - post_mean) ** 2

    is_alpha_credible = np.logical_and(
            scipy.stats.norm.ppf((1 - alpha) / 2) <= standard_y_test,
            standard_y_test <= scipy.stats.norm.ppf(1 - (1 - alpha) / 2)
        ).astype(float)

    log_lik = scipy.stats.norm.logpdf(x=y_test, loc=post_mean, scale=np.sqrt(post_var))

    row_metrics = pd.DataFrame({'row': row,
                                'output': [output] * y_test.shape[0],
                                'mse': mse,
                                'is_alpha_credible': is_alpha_credible,
                                'log_lik': log_lik,
                                'c_mean': [c_mean[0]] * y_test.shape[0],
                                'c_var': [c_var[0]] * y_test.shape[0],
                                'post_mean': post_mean,
                                'cost': [cost] * y_test.shape[0],
                                'status': [status] * y_test.shape[0],
                                'post_var': post_var,
                                'y_test': y_test}, columns=metrics.columns)

    for i in range(len(ls)):
        row_metrics['ls_dim_{}'.format(i+1)] = [ls[i]]*y_test.shape[0]
        assert (ls[i] > 0), "Estimated negative lengthscale"

    metrics = pd.concat((metrics, row_metrics), ignore_index=True)

    return metrics


def get_proper_estimates(
        metrics, data, predictors, outputs, alpha, in_sample, model):

    if in_sample:
        indexes = [0]
    else:
        indexes = range(data.shape[0])

    for i in indexes:
        if in_sample:
            data_train_loo = data
            data_test_loo = data
        else:
            data_train_loo = data[data['row'] != i]
            data_test_loo = data[data['row'] == i]
        x_train = data_train_loo[predictors].values
        x_test = data_test_loo[predictors].values
        for output in outputs:
            print(output)
            y_train = data_train_loo[[output]].values
            y_test = data_test_loo[[output]].values

            model.clean()

            # try:
            model.set_data(x_train, y_train)
            model.train()

            ls = model.get_ls()
            c_mean = model.get_c_mean()

            c_var = model.get_c_var()
            status = model.get_status()
            objective_value = model.get_objective_value()

            post_mean, post_var = model.predict(x_test)
            # except np.linalg.LinAlgError:
            #     ls = np.ones(x_train.shape[1])
            #     c_mean = np.array([np.nan])
            #     c_var = np.array([np.nan])
            #     status = 'Failed'
            #     objective_value = np.nan
            #     post_mean = np.nan * np.ones(y_test.shape)
            #     post_var = np.nan * np.ones(y_test.shape)

            if in_sample:
                metrics = update_metrics_with_posterior(
                 np.nan * np.ones(1), np.nan * np.ones(1),
                 np.nan * np.ones(1), metrics, alpha, output, [-1],
                 c_mean, ls, c_var, objective_value, status)
            else:
                metrics = update_metrics_with_posterior(
                 y_test.reshape(-1), post_mean.reshape(-1),
                 post_var.reshape(-1), metrics, alpha, output, [i], c_mean,
                 ls, c_var, objective_value, status)

    return metrics


def get_fixed_parameters_loo(metrics, data, predictors, outputs, alpha, model):
    for output in outputs:
        x_train = data[predictors].values
        y_train = data[[output]].values

        model.clean()

        model.set_data(x_train, y_train)
        model.train()

        ls = model.get_ls()
        c_mean = model.get_c_mean()

        c_var = model.get_c_var()
        status = model.get_status()
        objective_value = model.get_objective_value()

        model.set_loo_posterior_metrics()
        post_mean = model.get_loo_mean()
        post_var = model.get_loo_var()

        metrics = update_metrics_with_posterior(
         y_train.reshape(-1), post_mean.reshape(-1), post_var.reshape(-1),
         metrics, alpha, output, range(post_mean.shape[0]), c_mean, ls,
         c_var, objective_value, status)

    return metrics


def get_metrics(
        file, predictors, outputs, reestimate_param, alpha, in_sample, model):
    data = pd.read_csv(file, sep=',', index_col=0)
    data['row'] = range(data.shape[0])

    metrics = pd.DataFrame(
     columns=['row', 'output', 'mse', 'is_alpha_credible', 'log_lik']
     + ['c_mean', 'c_var', 'post_mean', 'cost', 'status', 'post_var', 'y_test']
     + ['ls_dim_{}'.format(x+1) for x in range(len(predictors))])

    if reestimate_param:
        metrics = get_proper_estimates(metrics, data, predictors, outputs,
                                       alpha, in_sample, model=model)
    else:
        metrics = get_fixed_parameters_loo(metrics, data, predictors, outputs,
                                           alpha, model=model)

    return metrics


def launch_data_set_optimization_evaluations(
        data_path, destination_path, model):

    data = pd.read_csv(data_path, sep=',', index_col=0)

    predictors = [a for a in data.columns if 'x' in a]
    outputs = [a for a in data.columns if 'x' not in a]

    metrics_in_sample = get_metrics(data_path, predictors, outputs,
                                    reestimate_param=True, alpha=0.95,
                                    in_sample=True, model=model)

    column_to_write = ['row', 'output', 'c_mean', 'c_var', 'status', 'cost'] \
        + [a for a in metrics_in_sample if 'ls_dim' in a]

    metrics_in_sample['row'] = -1
    full_sample = metrics_in_sample[column_to_write]

    target_dir = os.path.dirname(destination_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    full_sample.to_csv(destination_path)


def launch_full_optimization_evaluations(data_dir, destination_dir, models):
    for file in os.listdir(data_dir):
        print(file)
        for model in models:
            launch_data_set_optimization_evaluations(
                os.path.join(data_dir, file),
                os.path.join(destination_dir,
                             model[0], file), model[1])


def launch_data_set_perf_evaluations(data_path, destination_path, model):

    data = pd.read_csv(data_path, sep=',', index_col=0)

    predictors = [a for a in data.columns if 'x' in a]
    outputs = [a for a in data.columns if 'x' not in a]

    metrics_loo = get_metrics(data_path, predictors, outputs,
                              reestimate_param=True, alpha=0.95,
                              in_sample=False, model=model)

    target_dir = os.path.dirname(destination_path)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    metrics_loo.to_csv(destination_path)


def launch_full_perf_evaluations(data_dir, destination_dir, models):
    for file in os.listdir(data_dir):
        for model in models:
            print(file)
            launch_data_set_perf_evaluations(
             os.path.join(data_dir, file),
             os.path.join(destination_dir, model[0], file), model[1])
