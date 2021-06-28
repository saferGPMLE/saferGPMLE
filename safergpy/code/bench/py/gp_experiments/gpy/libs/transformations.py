# A thin wrapper around paramz.transformations

import numpy as np
import logging

import paramz.domains as domains

from paramz.transformations import Transformation

from paramz.transformations import (Logexp,
                                    Exponent,
                                    NegativeLogexp,
                                    NegativeExponent,
                                    Square,
                                    Logistic)

# Public API
__all__ = ['domains', 'IdentityPosReals', 'Logexp', 'Logistic',
           'Exponent', 'NegativeLogexp', 'NegativeExponent',
           'ReciprocalSquareRoot', 'Square']

logger = logging.getLogger(__name__)


class IdentityPosReals(Transformation):
    domain = domains._POSITIVE

    def f(self, x):
        return x

    def finv(self, x):
        return x

    def initialize(self, f):
        if np.any(f < 0.):
            logger.info("Warning: changing parameters to satisfy constraints")
        return np.abs(f)

    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, np.exp(f-f))

    def __str__(self):
        return '+ve'


class ReciprocalSquareRoot(Transformation):
    domain = domains._POSITIVE

    def f(self, x):
        return 1/np.sqrt(x)

    def finv(self, x):
        return 1/(x ** 2)

    def initialize(self, f):
        if np.any(f < 0.):
            logger.info("Warning: changing parameters to satisfy constraints")
        return np.abs(f)

    def gradfactor(self, f, df):
        return np.einsum('i,i->i', df, (-0.5) * (f ** (-1.5)))

    def __str__(self):
        return '+sq'
