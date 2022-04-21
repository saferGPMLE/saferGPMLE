def select_package(pkg):

    pkg = pkg.lower()

    if pkg == 'scikit-learn':
        from pythongp.wrappers import sklearn_wrapper
        pgp = sklearn_wrapper.sklearn_wrapper()
    elif pkg == 'gpy':
        from pythongp.wrappers import gpy_wrapper
        pgp = gpy_wrapper.gpy_wrapper()
    elif pkg == 'gpflow':
        from pythongp.wrappers import gpflow_wrapper
        pgp = gpflow_wrapper.gpflow_wrapper()
    elif pkg == 'gpytorch':
        from pythongp.wrappers import gpytorch_wrapper
        pgp = gpytorch_wrapper.gpytorch_wrapper()
    elif pkg == 'openturns':
        from pythongp.wrappers import openturns_wrapper
        pgp = openturns_wrapper.openturns_wrapper()
    elif pkg == 'pylibkriging':
        from pythongp.wrappers import pylibkriging_wrapper
        pgp = pylibkriging_wrapper.pylibkriging_wrapper()
    else:
        raise ValueError('Unexpected package name : {}'.format(pkg))
    return pgp
