'''
    Wrapper functions
    Author: S.B
    Description: The following code contains the necessary wrapper functions
    which implements Gaussian Process regression the GPy library
'''
import GPy
import numpy as np
import math


class gpy_wrapper():

    def __init__(self):

        # library
        self.library = 'gpy'

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
        self.z_train = np.reshape(z_train,(len(z_train),1))
        self.x_train = x_train
        self.input_dim = x_train.shape[1]


    def set_kernel(self, kernel, ard=True):
        '''
        kernel : dictionary of parameters
        '''

        if kernel['name'] == 'Matern':
            if kernel['order'] == 1.5:
                self.kernel_function = GPy.kern.Matern32(input_dim=self.input_dim,
                                                         variance=kernel['scale'],
                                                         lengthscale=kernel['lengthscale'],
                                                         ARD=ard)
            elif kernel['order'] == 2.5:
                self.kernel_function = GPy.kern.Matern52(input_dim=self.input_dim,
                                                         variance=kernel['scale'],
                                                         lengthscale=kernel['lengthscale'],
                                                         ARD=ard)

        elif kernel['name'] == 'Gaussian':
            self.kernel_function = GPy.kern.RBF(input_dim=self.input_dim,
                                                variance=kernel['scale'],
                                                lengthscale=kernel['lengthscale'],
                                                ARD=ard)
        else:
            self.kernel_function = "This library does not support the specified kernel function"


    def set_mean(self, mean):
        '''
        This function constructs the mean function
        '''

        if mean == 'constant':
            self.mean_function = GPy.mappings.constant.Constant(input_dim=self.x_train.shape[1], output_dim=1, value=0.0)
        elif mean != 'zero':
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

        else:
            self.model = GPy.models.GPRegression(self.x_train, self.z_train, kernel=self.kernel_function,
                                        Y_metadata=None, normalizer=None,
                                        noise_var=noise, mean_function=self.mean_function)

            if hasattr(self.model, 'sum'):
                self.model.sum.constant.variance.fix()

        print('\nBefore optimization : \n',self.model)
        
    
    def optimize(self, param_opt, itr):
        
        if param_opt in ['MLE', 'MLE_with_smart_init']:
            optimizer_input = True
            if param_opt == 'MLE':
                #if itr <= 1:
                    self.model.optimize(messages=optimizer_input, max_iters=1000, start=None, clear_after_finish=False,
                                   ipython_notebook=True)
                #else:
                    #self.model.optimize_restarts(num_restarts=itr)
            else:
                grid = np.vectorize(lambda x: math.exp(x * math.log(10)))(np.arange(-5, 5, 1))
                scores = []

                zero_mean = not hasattr(self.model, 'constmap')

                for ls in grid:
                    self.set_isotropic_lengthscale(ls)

                    beta, variance = self.get_beta_and_var_from_ls(zero_mean, hasattr(self.model, 'sum'))

                    self.set_var(variance)

                    self.set_beta(beta, zero_mean)

                    scores.append(self.model._objective_grads(self.model.optimizer_array)[0])

                print("grid : {}".format(grid))
                print("scores : {}".format(scores))

                best_model_index = np.argmin(scores)

                self.set_isotropic_lengthscale(grid[best_model_index])

                beta, variance = self.get_beta_and_var_from_ls(zero_mean, hasattr(self.model, 'sum'))

                self.set_var(variance)

                self.set_beta(beta, zero_mean)

                self.model.optimize(messages=optimizer_input, max_iters=1000, start=None, clear_after_finish=False,
                               ipython_notebook=True)

            print('\nAfter optimization : \n', self.model)

            if hasattr(self.model, 'sum'):
                path = self.model.sum
            else:
                path = self.model
            if hasattr(path, 'Mat52'):
                lengthscales = path.Mat52.lengthscale
            if hasattr(path, 'Mat32'):
                lengthscales = path.Mat32.lengthscale
            if hasattr(path, 'rbf'):
                lengthscales = path.rbf.lengthscale

            print("values : {} ".format(lengthscales))
            print("\nOptimized parameters\n", self.model.param_array)

        elif param_opt != 'Not_optimize':
            return ("Not sure whether this library supports the specified Parameter optimizer")

    
    def predict(self, x_test):
        '''
        This function makes predictions for the test data
        '''

        self.x_test = x_test
        
        if type(self.model) == str:
            return
        
        self.z_postmean, self.z_postvar = self.model.predict(self.x_test, include_likelihood=True)

        return self.z_postmean.reshape(-1), np.sqrt(self.z_postvar.reshape(-1))

