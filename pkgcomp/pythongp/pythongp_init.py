def select_library(library):

    if library == 'Sklearn':
        from pythongp.wrappers import sklearn_wrapper
        pgp = sklearn_wrapper.sklearn_wrapper()
    elif library == 'GPy':
        from pythongp.wrappers import gpy_wrapper
        pgp = gpy_wrapper.gpy_wrapper()
    elif library == 'GPflow':
        from pythongp.wrappers import gpflow_wrapper
        pgp = gpflow_wrapper.gpflow_wrapper()
    elif library == 'GPytorch':
        from pythongp.wrappers import gpytorch_wrapper
        pgp = gpytorch_wrapper.gpytorch_wrapper()
    elif library == 'ot':
        from pythongp.wrappers import openturns_wrapper
        pgp = openturns_wrapper.openturns_wrapper()    
    else:
        raise ValueError('Unexpected library name : {}'.format(library))
    return pgp

