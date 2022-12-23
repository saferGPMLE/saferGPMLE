'''
    Wrapper functions
    Author: Y.Richet
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the pylibkriging library
'''
import numpy as np
import math
import pylibkriging as lk

class pylibkriging_wrapper():

    def __init__(self):

        # library
        self.library = 'pylibkriging'

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

    def load_data(self, x_train, z_train):
        '''
        This function re-configures the training data according to the library requirement
        '''
        self.z_train = np.reshape(z_train, (len(z_train)))
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
            self.mean_function = mean
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
            self.mean_function = 'constant'

        self.model = lk.Kriging(self.kernel_function)

        print('\nBefore optimization : \n', self.model.summary())

    def optimize(self, param_opt, itr):

        if param_opt in ['MLE']:
            if param_opt == 'MLE':
                self.model.fit(self.z_train, self.x_train, 
                "constant", False, 
                "BFGS10", "LL", {})
                
            print('\nAfter optimization : \n', self.model.summary())

            lengthscales = np.transpose(self.model.theta())

            print("values : {} ".format(lengthscales))

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")

    def get_NLL(self):
        return -self.model.logLikelihood()

    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''

        self.x_test = x_test

        if type(self.model) == str:
            return
        
        pred = self.model.predict(np.array(self.x_test), True, False, False)
        self.z_postmean = pred[0][:,0]
        self.z_postvar = pred[1][:,0]**2

        return self.z_postmean, np.sqrt(self.z_postvar)
