'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the OpenTURNS library
'''
import openturns as ot
import numpy as np
import math


class openturns_wrapper():

    def __init__(self):

        # library
        self.library = 'openturns'

        # model definition
        self.model = None

        self.mean_function = None

        self.kernel_function = None

        self.nugget = None

        # data
        self.input_dim = 1
        self.train_dataframe = None
        self.x_train = None
        self.z_train = None

        self.test_dataframe = None
        self.x_test = None
        self.z_postmean = None
        self.z_postvar = None

    def load_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.x_train = ot.Sample(x_train)
        self.z_train = ot.Sample(np.reshape(z_train, (len(self.x_train), 1)))
        self.input_dim = x_train.shape[1]

    def set_kernel(self, kernel, ard=True):
        '''
        kernel : dictionary of parameters
        '''

        if kernel['name'] == 'Matern':
            self.kernel_function = ot.MaternModel(
                kernel['lengthscale'],
                [math.sqrt(kernel['variance'])],
                float(kernel['order']))
        elif kernel['name'] == 'Gaussian':
            self.kernel_function = ot.SquaredExponential(
                kernel['lengthscale'],
                [math.sqrt(kernel['variance'])])
        else:
            self.kernel_function = "This library does not support the specified kernel function"

    def set_mean(self, mean):
        '''
        This function constructs the mean function
        '''

        if mean == 'constant':
            self.mean_function = ot.ConstantBasisFactory(self.input_dim).build()
        elif mean == 'zero':
            self.mean_function = ot.Basis()
        else:
            self.mean_function = "This library does not support the specified mean function"

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

        self.nugget = noise
        self.kriging_algorithm = ot.KrigingAlgorithm(
            self.x_train, self.z_train, self.kernel_function, self.mean_function)
        self.kriging_algorithm.setNoise([self.nugget] * len(self.x_train))

    def optimize(self, param_opt, itr):

        if param_opt == 'MLE':
            self.kriging_algorithm.setOptimizeParameters(optimizeParameters=True)
        elif param_opt == 'Not_optimize':
            self.kriging_algorithm.setOptimizeParameters(optimizeParameters=False)
        else:
            return ("This library does not support the specified Parameter optimizer")

        print(self.kernel_function.getFullParameterDescription())
        print("parameter before optimization : ", self.kernel_function.getFullParameter())

        self.kriging_algorithm.run()

        self.model = self.kriging_algorithm.getResult()
        print("parameter after optimization : \n", self.model.getCovarianceModel())
        print("Nugget", self.kriging_algorithm.getNoise())

    def get_NLL(self):
        lik_function = self.kriging_algorithm.getReducedLogLikelihoodFunction()
        NLL = -lik_function(self.model.getCovarianceModel().getScale())
        return NLL[0]

    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''

        self.x_test = ot.Sample(x_test)

        if type(self.model) == str:
            return

        self.z_postmean = np.array(self.model.getConditionalMean(self.x_test))
        self.z_postvar = np.sqrt(np.add(np.diag(
            np.array(self.model.getConditionalCovariance(self.x_test))), self.nugget))

        return self.z_postmean, self.z_postvar
