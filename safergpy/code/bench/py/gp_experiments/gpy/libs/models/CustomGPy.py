import os, sys

import GPy
import libs.utils.gpy_estimation_lib as gpy_estimation_lib
import libs.utils.initialisor as initialisor
import libs.transformations as transfo
import numpy as np
import scipy.stats
import math
from scipy.stats import norm


class CustomGPy():
    eps = 0.001
    is_zero_mean = False

    __slots__ = ['lengthscale_constraint', 'variance_constraint',
                 'init', 'optim_opts', 'kernel_function', 'mean_function',
                 'model', 'status', '_input_values', '_output_values', 'loo_mean',
                 'loo_var', 'MGE', 'AGE', 'untrained_variance', 'untrained_lengthscale',
                 'fix_noise', 'fix_var', 'profiler', 'do_profiling', 'postvar_options',
                 '_input_ordinates', '_input_factors', '_output_ordinate', '_output_factor',
                 'input_transform_type', 'output_transform_type']

    def __init__(self,
                 lengthscale_constraint_class=transfo.Exponent,
                 variance_constraint_class=transfo.Exponent,
                 init="classic", stopping_criterion="strict",
                 optim_scheme=[[10, 3.0], [2, 1.0], [2, 1.0]],
                 untrained_variance=1, untrained_lengthscale=1,
                 fix_noise=True, fix_var=False,
                 profiler=gpy_estimation_lib.analytical_mean_and_variance_optimization,
                 postvar_options={"value": 0, "type": 'Error'},
                 input_transform_type='Hypercube',
                 output_transform_type='Standardize',
                 do_profiling=True
                 ):

        self.input_transform_type = input_transform_type
        self.output_transform_type = output_transform_type

        self.untrained_variance = untrained_variance
        self.untrained_lengthscale = untrained_lengthscale

        self.set_parametrization(
            lengthscale_constraint_class,
            variance_constraint_class,
        )

        if init in ['brutal', 'classic', 'scaled_anisotropic_init',
                    'classic_profiled', 'custom']:
            self.init = init
        else:
            raise ValueError('Unknown method : {}'.format(init))

        assert not (fix_var and profiler is not None)

        self.set_optim_opts(
            stopping_criterion=stopping_criterion,
            optim_scheme=optim_scheme,
            do_profiling=do_profiling,
            profiler=profiler
        )

        assert fix_noise, 'Not implemented yet'

        self.fix_noise = fix_noise
        self.fix_var = fix_var
        self.postvar_options = postvar_options

        self.kernel_function = None
        self.mean_function = None
        self.model = None
        self._input_values = None
        self._output_values = None
        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None
        self.status = None

    # --------------------------------------------------------------------------
    def set_optim_opts(self, stopping_criterion, optim_scheme, do_profiling, profiler):
        if stopping_criterion == 'strict':
            gtol = 10 ** (-20)
            bfgs_factor = 10
        elif stopping_criterion == 'soft':
            gtol = None
            bfgs_factor = None
        elif stopping_criterion == 'intermediate':
            gtol = 10 ** (-14)
            bfgs_factor = 10
        else:
            raise ValueError("Unknown stopping criterion setting : {}.".format(stopping_criterion))

        self.profiler = profiler
        self.do_profiling = do_profiling

        self.optim_opts = {
            'optim_scheme': optim_scheme,
            'gtol': gtol,
            'bfgs_factor': bfgs_factor
        }

    # --------------------------------------------------------------------------
    def set_parametrization(
            self,
            lengthscale_constraint_class,
            variance_constraint_class,
        ):

        self.lengthscale_constraint = lengthscale_constraint_class()
        assert (self.lengthscale_constraint.domain == transfo.domains._POSITIVE)

        self.variance_constraint = variance_constraint_class()
        assert (self.variance_constraint.domain == transfo.domains._POSITIVE)

    # --------------------------------------------------------------------------
    def set_kernel(self):

        # Create a kernel function object with default parametrizations
        self.kernel_function\
            = GPy.kern.Matern52(input_dim=self._input_values.shape[1],
                                variance=self.untrained_variance,
                                lengthscale=self.untrained_lengthscale, ARD=True)

        # Set parametrization and/or constraints for the range parameters
        self.kernel_function\
            .lengthscale.constrain(transform=self.lengthscale_constraint)

        # Set parametrization and/or constraints for the variance parameter
        self.kernel_function\
            .variance.constrain(transform=self.variance_constraint)

        if self.fix_var:
            self.kernel_function.variance.fix()

    def transform_input(self, x):
        tiled_ordinates = np.tile(self._input_ordinates, reps=[x.shape[0], 1])
        tiled_factors = np.tile(self._input_factors, reps=[x.shape[0], 1])

        assert x.shape == tiled_ordinates.shape and x.shape == tiled_factors.shape

        return (x - tiled_ordinates)/tiled_factors

    def scale_input_back(self, x):
        assert x.shape == self._input_factors.shape
        return x * self._input_factors

    def transform_output(self, y):
        assert isinstance(self._output_ordinate, float) and isinstance(self._output_factor, float)

        return (y - self._output_ordinate)/self._output_factor

    def transform_post_mean(self, y):
        assert isinstance(self._output_ordinate, float) and isinstance(self._output_factor, float)

        return y*self._output_factor + self._output_ordinate

    def transform_post_var(self, var):
        assert isinstance(self._output_factor, float)

        return (self._output_factor**2) * var

    # ---------------------------------------------------------------------------
    def set_data(self, input_data, output_data):
        assert isinstance(input_data, np.ndarray) and isinstance(output_data, np.ndarray)
        assert input_data.ndim == 2 and output_data.ndim == 2
        assert input_data.shape[0] == output_data.shape[0] and output_data.shape[1] == 1

        if self._input_values is not None:
            if input_data.shape[1] != self._input_values.shape[1]:
                print('Warning : X dimensionality differs from original data. Cleaning the model.')
                self.clean()

        if self._input_values is None:
            self.store_normalizers(input_data, output_data)

        self._input_values = self.transform_input(input_data)
        self._output_values = self.transform_output(output_data)

        self.check_training_type()

        if self.model is None:
            self.re_build_model()
        else:
            self.repair_model()

    # ---------------------------------------------------------------------------
    def store_normalizers(self, input_data, output_data):
        if self.input_transform_type == 'Hypercube':
            self._input_ordinates = input_data.min(0)
            self._input_factors = input_data.max(0) - input_data.min(0)
        elif self.input_transform_type == 'None':
            self._input_ordinates = np.zeros(input_data.shape[1])
            self._input_factors = np.ones(input_data.shape[1])
        elif self.input_transform_type == 'Standardize':
            self._input_ordinates = input_data.mean(0)
            self._input_factors = input_data.std(0)
        else:
            raise ValueError(self.input_transform_type)

        if self.output_transform_type == 'Standardize':
            self._output_ordinate = output_data.mean()
            self._output_factor = output_data.std()
        elif self.output_transform_type == 'None':
            self._output_ordinate = 0.0
            self._output_factor = 1.0
        else:
            raise ValueError(self.output_transform_type)

    # ---------------------------------------------------------------------------
    def re_build_model(self):
        self.check_training_type()

        self.mean_function = GPy.mappings.constant.Constant(input_dim=self._input_values.shape[1], output_dim=1, value=0.0)

        self.set_kernel()

        self.repair_model()

        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None
        self.status = "Untrained"

    # ---------------------------------------------------------------------------
    def repair_model(self):
        self.check_training_type()

        self.model = GPy.models.GPRegression(self._input_values, self._output_values, kernel=self.kernel_function,
                                             Y_metadata=None, normalizer=None,
                                             noise_var=0, mean_function=self.mean_function)

        if self.fix_noise:
            self.model.Gaussian_noise.variance.fix()

        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None
        self.status = "Untrained"

    # ---------------------------------------------------------------------------
    def initialize(self):
        self.check_training_type()

        init_opts = {'fix_var': self.fix_var, 'profiler': self.profiler, 'is_zero_mean': self.is_zero_mean}

        if self.init == 'scaled_anisotropic_init':
            self.model = initialisor.grid_init(self.model, isotropic=False, **init_opts)
        elif self.init == 'classic':
            self.model = initialisor.std_init(
                self.model,
                use_dimension=True,
                profiler=None,
                fix_var=self.fix_var,
                is_zero_mean=self.is_zero_mean
            )
        elif self.init == 'classic_profiled':
            self.model = initialisor.std_init(self.model, use_dimension=True, **init_opts)
        elif self.init in ['custom', 'brutal']:
            pass
        else:
            raise NotImplementedError('{} init method'.format(self.init))

    # ---------------------------------------------------------------------------
    def train(self):
        if self.init == 'brutal':
            self.model, self.status = gpy_estimation_lib.brutal_train(
                self.model,
                n=self.optim_opts['optim_scheme'][0][0],
                profiler=self.profiler
            )
        else:
            self.initialize()

            self.check_training_type()

            assert not (self.fix_var and self.profiler is not None)

            if self.do_profiling:
                trainer_profiler = self.profiler
            else:
                trainer_profiler = None

            self.model, self.status = gpy_estimation_lib.trainer(
                self.model,
                options=self.optim_opts,
                profiler=trainer_profiler
            )

        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None

    # ---------------------------------------------------------------------------
    def predict(self, data):
        self.check_testing_type(data)

        y_pred, var = self.model.predict(self.transform_input(data))

        if np.any(var < self.postvar_options['value']):
            if self.postvar_options['type'] == 'Error':
                raise ValueError("Variance below threshold : {}".format(self.postvar_options['value']))
            elif self.postvar_options['type'] == 'truncate':
                var[var < self.postvar_options['value']] = self.postvar_options['value']
            elif self.postvar_options['type'] == 'None':
                pass
            else:
                raise ValueError(self.postvar_options['type'])

        return self.transform_post_mean(y_pred), self.transform_post_var(var)

    # ---------------------------------------------------------------------------
    def get_cdf(self, x, data):
        y_pred, y_var = self.predict(data)

        assert y_pred.shape == y_var.shape, "Shape issue"
        assert isinstance(x, float), "x must be float"

        y_sd = np.vectorize(math.sqrt)(y_var)

        return scipy.stats.norm.cdf((x - y_pred)/y_sd)

    # ---------------------------------------------------------------------------
    def get_y_quantiles(self, q, data):
        y_pred, y_var = self.predict(data)

        assert y_pred.shape == y_var.shape, "Shape issue"
        assert isinstance(q, float), "x must be float"

        return norm.ppf(q, loc=y_pred, scale=np.vectorize(math.sqrt)(y_var))

    # ---------------------------------------------------------------------------
    def get_gaussian_normalized(self, data, truth):
        y_pred, y_var = self.predict(data)

        assert truth.shape == y_var.shape and truth.shape == y_pred.shape, "Shape issue"

        return (truth - y_pred) / np.vectorize(math.sqrt)(y_var)

    # ---------------------------------------------------------------------------
    def sample_y(self, data, n_samples=1):
        self.check_testing_type(data)

        y_pred = self.model.posterior_samples(X=self.transform_input(data), size=n_samples)

        return self.transform_post_mean(y_pred)

    # ---------------------------------------------------------------------------
    def get_loo_mean(self):
        if self.loo_mean is None:
            self.set_loo_posterior_metrics()
        return self.loo_mean

    # ---------------------------------------------------------------------------
    def get_loo_var(self):
        if self.loo_var is None:
            self.set_loo_posterior_metrics()
        return self.loo_var

    # ---------------------------------------------------------------------------
    def get_age(self):
        if self.AGE is None:
            self.set_loo_posterior_metrics()
        return self.AGE

    # ---------------------------------------------------------------------------
    def get_mge(self):
        if self.MGE is None:
            self.set_loo_posterior_metrics()
        return self.MGE

    # ---------------------------------------------------------------------------
    def set_loo_posterior_metrics(self):
        g = self.model.posterior.woodbury_vector
        c = self.model.posterior.woodbury_inv
        y = self.model.Y_normalized

        c_diag = np.diag(c)[:, None]

        assert isinstance(g, np.ndarray) and isinstance(c_diag, np.ndarray) \
               and isinstance(y, np.ndarray), 'Type issue'
        assert g.shape == c_diag.shape and y.shape == g.shape, "Shape issue"

        mu = y - g / c_diag
        var = 1 / c_diag

        self.loo_mean = mu
        self.loo_var = var

        assert self._output_values.shape == self.loo_mean.shape, "Shape issue"

        self.AGE = 100 * np.mean(
            abs(self.loo_mean - self._output_values) / (self._output_values.max() - self._output_values.min()))
        self.MGE = 100 * np.max(
            abs(self.loo_mean - self._output_values) / (self._output_values.max() - self._output_values.min()))

    # ---------------------------------------------------------------------------
    def check_training_type(self):
        if not (isinstance(self._input_values, np.ndarray) and isinstance(self._output_values, np.ndarray)):
            raise TypeError("Input and output values should be numpy arrays. They are respectively {}, {}".format(
                type(self._input_values), type(self._output_values)))

    def check_testing_type(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input and output values should be numpy arrays, not {}".format(type(data)))

        assert data.shape[1] == self._input_values.shape[1]

    # ---------------------------------------------------------------------------
    def clean(self):
        self.model = None
        self.kernel_function = None
        self.mean_function = None
        self._input_values = None
        self._output_values = None
        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None
        self.status = "Untrained"

        self.clean_normalization()

    # ---------------------------------------------------------------------------
    def clean_normalization(self):
        self._input_ordinates = None
        self._input_factors = None

        self._output_ordinate = None
        self._output_factor = None

    # ---------------------------------------------------------------------------
    def __str__(self):
        if self.model is not None:
            return(self.model.__str__() + "\n" + self.model.kern.lengthscale.__str__())
        else:
            return("Model unset for now.")

    # ---------------------------------------------------------------------------
    def get_c_mean(self):
        if self.model is None:
            raise ValueError("Model is None, data probably hasnt been defined.")
        else:
            return self.transform_post_mean(self.model.constmap.C.values.copy())

    # ---------------------------------------------------------------------------
    def get_c_var(self):
        if self.model is None:
            raise ValueError("Model is None, data probably hasnt been defined.")
        else:
            return self.transform_post_var(self.model.kern.variance.values.copy())

    # ---------------------------------------------------------------------------
    def get_ls(self):
        if self.model is None:
            raise ValueError("Model is None, data probably hasnt been defined.")
        else:
            std_ls = self.model.kern.lengthscale.values.copy()

            return self.scale_input_back(std_ls)

    # # ---------------------------------------------------------------------------
    # def set_ls(self, value):
    #     if self.model is None:
    #         raise ValueError("Model is None, data probably hasnt been defined.")
    #     else:
    #         gpy_estimation_lib.set_gpy_model_ls(self.model, value)

    # ---------------------------------------------------------------------------
    def get_status(self):
        if self.status is None:
            raise ValueError("Status is None, data probably hasnt been defined.")
        else:
            return self.status

    # ---------------------------------------------------------------------------
    def get_objective_value(self):
        if self.model is None:
            raise ValueError("Model is None, data probably hasnt been defined.")
        else:
            return self.model.objective_function() + 0.5*self._output_values.shape[0]*math.log(self._output_factor**2)

    # ---------------------------------------------------------------------------
    def get_rkhs_semi_norm(self):
        raise NotImplementedError("Be carefull, m may not be the IK's one")

    # ---------------------------------------------------------------------------
    def plot_warping(self):
        pass

    # ---------------------------------------------------------------------------
    def to_dict(self):
        raise NotImplementedError
        # model_dict = {}
        # for k in ['param', 'init', 'gtol', 'bfgs_factor', 'do_restarts', 'restart_sessions_limit', 'num_restarts',
        #          'analytical_mu_and_sigma2_optimization', 'end_analytical_mu_and_sigma2_optimization',
        #          'status']:
        #     model_dict[k] = self.__getattribute__(k)
        #
        # model_dict['model'] = self.model.to_dict()
        #
        # return model_dict

    # ---------------------------------------------------------------------------
    def set_model_from_dict(self, d):
        raise NotImplementedError
        # for k in ['param', 'init', 'gtol', 'bfgs_factor', 'do_restarts', 'restart_sessions_limit', 'num_restarts',
        #          'analytical_mu_and_sigma2_optimization', 'end_analytical_mu_and_sigma2_optimization',
        #          'status']:
        #     self.__setattr__(k, d[k])
        #
        # self.model = GPy.models.GPRegression.from_dict(d['model'])
        #
        # self.kernel_function = self.model.kern
        # self.mean_function = self.model.mean_function
        # self._input_values = self.model.X
        # self._output_values = self.model.Y
        # self.set_loo_posterior_metrics()

    # ---------------------------------------------------------------------------
    @staticmethod
    def default_args():
        return {'init': "classic",
                'stopping_criterion': "strict",
                'do_profiling': True,
                'do_restarts': True,
                'do_end_profiling': True,
                'n_multistarts': 2,
                'n_iterations': 3}

    # ---------------------------------------------------------------------------
    @staticmethod
    def default_setup():
        return CustomGPy(**CustomGPy.default_args())
