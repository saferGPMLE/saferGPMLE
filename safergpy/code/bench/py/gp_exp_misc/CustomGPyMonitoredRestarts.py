import os, sys

from gp_experiments.gpy.libs.models.CustomGPy import CustomGPy
import GPy
import gp_exp_misc.gpy_estimation_lib_restarts_experiments as gpy_estimation_lib_restarts_experiments
import libs.transformations as transfo
import numpy as np
import scipy.stats
import math
from scipy.stats import norm

class CustomGPyMonitoredRestarts(CustomGPy):

    __slots__ = CustomGPy.__slots__ + ['bench_type']

    def __init__(self, **kwargs):

        if 'bench_type' not in kwargs.keys():
            self.bench_type = 'single'
        else:
            assert kwargs['bench_type'] in ['single', 'monte-carlo']
            self.bench_type = kwargs['bench_type']

            del kwargs['bench_type']

        if 'input_transform_type' not in kwargs.keys():
            kwargs['input_transform_type'] = 'None'
        if 'output_transform_type' not in kwargs.keys():
            kwargs['output_transform_type'] = 'None'

        super(CustomGPyMonitoredRestarts, self).__init__(**kwargs)

    # ---------------------------------------------------------------------------
    def train(self):

        if self.init == 'brutal':
            self.model, self.status = gpy_estimation_lib_restarts_experiments.brutal_train(
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

            l = gpy_estimation_lib_restarts_experiments.trainer_all(
                self.model,
                options=self.optim_opts,
                profiler=trainer_profiler,
                bench_type=self.bench_type
            )

            return l

        self.loo_mean = None
        self.loo_var = None
        self.MGE = None
        self.AGE = None
