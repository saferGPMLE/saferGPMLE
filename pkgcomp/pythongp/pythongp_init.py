def select_package(pkg):

    if pkg == 'Sklearn':
        from pythongp.wrappers import sklearn_wrapper
        pgp = sklearn_wrapper.sklearn_wrapper()
    elif pkg == 'GPy':
        from pythongp.wrappers import gpy_wrapper
        pgp = gpy_wrapper.gpy_wrapper()
    elif pkg == 'GPflow':
        from pythongp.wrappers import gpflow_wrapper
        pgp = gpflow_wrapper.gpflow_wrapper()
    elif pkg == 'GPytorch':
        from pythongp.wrappers import gpytorch_wrapper
        pgp = gpytorch_wrapper.gpytorch_wrapper()
    elif pkg == 'ot':
        from pythongp.wrappers import openturns_wrapper
        pgp = openturns_wrapper.openturns_wrapper()    
    else:
        raise ValueError('Unexpected package name : {}'.format(pkg))
    return pgp

