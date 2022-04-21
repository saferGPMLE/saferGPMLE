'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the DiceKriging library
'''
import numpy as np
import math

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

class dicekriging_wrapper():

    def __init__(self):

        # library
        self.library = 'dicekriging'

        # model definition
        self.model = None

        self.mean_function = None

        self.kernel_function = None

        # data
        self.input_dim = 1
        self.train_dataframe = None
        self.x_train = None
        self.z_train = None

        self.test_dataframe = None
        self.x_test = None
        self.z_postmean = None
        self.z_postvar = None

        importr('DiceKriging') # so later, we can call ro['km'] or ro.km

    def load_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.z_train = np.reshape(z_train, (len(z_train), 1))
        self.x_train = x_train
        self.input_dim = x_train.shape[1]

    def set_kernel(self, kernel, ard=True):
        '''
        kernel : dictionary of parameters
        '''

        if kernel['name'] == 'Matern':
            if kernel['order'] == 1.5:
                self.kernel_function = 'matern3_2'

            elif kernel['order'] == 2.5:
                self.kernel_function = 'matern5_2'

        elif kernel['name'] == 'Gaussian':
            self.kernel_function = 'gauss'

        else:
            self.kernel_function = "This library does not support the specified kernel function"

    def set_mean(self, mean):
        '''
        This function constructs the mean function
        '''

        if mean == 'constant':
            self.mean_function = '~1'
        elif mean != 'zero':
            self.mean_function = ("Not sure whether this library "
                                  "supports the specified mean function")

    def init_model(self, noise):
        '''
        This function constructs the regression model
        '''
        if type(self.kernel_function) == str or type(self.mean_function) == str:
            if type(self.kernel_function) == str:
                print(self.kernel_function)
            if type(self.mean_function) == str:
                print(self.mean_function)
            self.model = 'No model'

        if self.mean_function is None:
            self.mean_function = '~1'

        self.model = ro.r.km(
            design=ro.r.matrix(self.x_train,nrow=self.x_train.shape[0]), response=ro.r.matrix(self.z_train), covtype=self.kernel_function,
            formula=ro.r['as.formula'](self.mean_function), 
            control=ro.r.list(maxit=0), # to early-stop optimization...
            **{'noise.var':ro.r.rep(noise,self.x_train.shape[0])}
            )

        print('\nBefore optimization : \n', self.model)

    def optimize(self, param_opt, itr):

        if param_opt in ['MLE']:
            if param_opt == 'MLE':
                self.model = ro.r.km(
                    design=ro.r.matrix(self.x_train,nrow=self.x_train.shape[0]), response=ro.r.matrix(self.z_train), covtype=self.kernel_function,
                    formula=ro.r['as.formula'](self.mean_function))
                
            print('\nAfter optimization : \n', self.model)

            cov = self.model.slots['covariance']
            lengthscales = cov.slots['range.val']

            print("values : {} ".format(lengthscales))

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")

    def get_NLL(self):
        return -self.model.slots['logLik']

    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''

        self.x_test = x_test

        if type(self.model) == str:
            return

        pred = ro.r.predict(self.model, ro.r.matrix(self.x_test,nrow=self.x_test.shape[0]), type="UK", checkNames=False, **{'light.return':True})
        # print(ro.r.names(pred)) -> "trend"   "mean"    "sd"      "lower95" "upper95"
        self.z_postmean = pred[1]
        self.z_postvar = pred[2]**2

        return self.z_postmean, np.sqrt(self.z_postvar)
