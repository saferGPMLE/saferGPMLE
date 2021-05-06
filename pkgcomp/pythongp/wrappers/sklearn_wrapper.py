'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the Scikit learn library
'''
from ast import literal_eval as make_tuple
import sklearn.gaussian_process as sklearn_gp
import numpy as np


class sklearn_wrapper():

    def __init__(self):

        # library
        self.library = 'sklearn'

        # model definition
        self.model = None

        self.nugget = None

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

        self.z_train = z_train
        self.x_train = x_train
        self.input_dim = x_train.shape[1]

    def set_kernel(self, kernel, ard):
        '''
        kernel : dictionary of parameters
        '''

        if kernel['name'] == 'Matern':
            self.kernel_function = (
                sklearn_gp.kernels.Matern(
                    length_scale=kernel['lengthscale'],
                    nu=kernel['order'],
                    length_scale_bounds=make_tuple(kernel['lengthscale_bounds']))
                * sklearn_gp.kernels.ConstantKernel(
                    constant_value=kernel['variance']))
        elif kernel['name'] == 'Gaussian':
            self.kernel_function = (
                sklearn_gp.kernels.RBF(
                    length_scale=kernel['lengthscale'],
                    length_scale_bounds=make_tuple(kernel['lengthscale_bounds']))
                * sklearn_gp.kernels.ConstantKernel(
                    constant_value=kernel['variance']))
        else:
            self.kernel_function = "This library does not support the specified kernel function"

    def set_mean(self, mean):
        '''
        This function constructs the mean function
        '''

        if mean == 'constant':
            self.mean_function = True
        elif mean == 'zero':
            self.mean_function = False
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

    def optimize(self, param_opt, itr):

        if param_opt == 'MLE':
            optimizer_input = 'fmin_l_bfgs_b'
        elif param_opt == 'Not_optimize':
            optimizer_input = None
        else:
            return ("This library does not support the specified Parameter optimizer")

        self.model = sklearn_gp.GaussianProcessRegressor(
            kernel=self.kernel_function, alpha=self.nugget, optimizer=optimizer_input,
            normalize_y=self.mean_function, copy_X_train=True, random_state=None)
        print('Kernel hyperparameters before optimization :\n', self.model.kernel)
        self.model.fit(self.x_train, self.z_train)
        print('Kernel hyperparameters after optimization :\n', self.model.kernel_)
        print("Nuggets before optimization :\n", self.model.alpha)
        print("Likelihoood after optimization :\n", self.model.log_marginal_likelihood_value_)

    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''
        self.x_test = x_test

        if type(self.model) == str:
            return

        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, return_std=True)
        # To predict with the noise we need to add the likelihood variance to
        # the predicted posterior variance and take squareroot
        self.z_postvar = np.sqrt(np.add(np.square(self.z_postvar), self.nugget))

        return self.z_postmean, self.z_postvar
