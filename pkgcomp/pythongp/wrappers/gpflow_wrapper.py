'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the GPflow library
'''
import numpy as np
import gpflow


class gpflow_wrapper():

    def __init__(self):

        # library
        self.library = 'gpflow'

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
        self.z_train = np.reshape(z_train, (len(z_train), 1))
        self.x_train = x_train
        self.input_dim = x_train.shape[1]

    def set_kernel(self, kernel, ard=True):
        '''
        kernel : dictionary of parameters
        '''

        if kernel['name'] == 'Matern':
            if kernel['order'] == 1.5:
                self.kernel_function = gpflow.kernels.Matern32(input_dim=self.input_dim,
                                                               variance=kernel['scale'],
                                                               lengthscales=kernel['lengthscale'],
                                                               ARD=ard)
            elif kernel['order'] == 2.5:
                self.kernel_function = gpflow.kernels.Matern52(input_dim=self.input_dim,
                                                               variance=kernel['scale'],
                                                               lengthscales=kernel['lengthscale'],
                                                               ARD=ard)
            elif kernel['order'] == 0.5:
                self.kernel_function = gpflow.kernels.Matern12(input_dim=self.input_dim,
                                                               variance=kernel['scale'],
                                                               lengthscales=kernel['lengthscale'],
                                                               ARD=ard)
        elif kernel['name'] == 'Gaussian':
            self.kernel_function = gpflow.kernels.RBF(input_dim=self.input_dim,
                                                      variance=kernel['scale'],
                                                      lengthscales=kernel['lengthscale'],
                                                      ARD=ard)
        else:
            self.kernel_function = "This library does not support the specified kernel function"

    def set_mean(self, mean):
        '''
        This function constructs the mean function
        '''

        if mean == 'constant':
            self.mean_function = gpflow.mean_functions.Constant(c=np.ones(1)*np.mean(self.train_dataframe['z_train']))
        elif mean == 'zero':
            self.mean_function = gpflow.mean_functions.Zero()
        else:
            self.mean_function = "Not sure whether this library supports the specified mean function"

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

        else :

            self.model = gpflow.models.GPR(self.x_train, self.z_train, kern=self.kernel_function,
                                           mean_function=self.mean_function)
            self.model.likelihood.variance = noise

            print(self.model.as_pandas_table())

    def optimize(self, param_opt, itr):

        if param_opt == 'MLE':
            gpflow.train.ScipyOptimizer().minimize(self.model, disp=True)
            print('\n\nprintin AFTER optimization {}'.format(self.model.likelihood))

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")

        print(self.model.as_pandas_table())

    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''

        self.z_postmean, self.z_postvar = self.model.predict_y(x_test)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))
