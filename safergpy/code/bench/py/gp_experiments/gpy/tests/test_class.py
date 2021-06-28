import numpy as np
import scipy.integrate
import scipy
from libs.utils.scores import \
    Mse, SideLessMse, StandardizedMse, LogPredDensity, MinusCRPS
from libs.utils.metrics_computations import \
    get_crps, get_mse, get_gaussian_log_lik


n = 10000

np.random.seed(0)

y = np.random.uniform(size=[n])
mu_test = np.random.uniform(size=[n])
var_test = 2 + np.random.uniform(size=[n])

print("mse : {}".format(Mse().gradient(y=y, mu_pred=mu_test, var_pred=var_test)))
print("sidelessmse : {}".format(SideLessMse().gradient(y=y, mu_pred=mu_test, var_pred=var_test)))
print("StandardizedMse : {}".format(StandardizedMse().gradient(y=y, mu_pred=mu_test, var_pred=var_test)))
print("LogPredDensity : {}".format(LogPredDensity().gradient(y=y, mu_pred=mu_test, var_pred=var_test)))
print("MinusCRPS : {}".format(MinusCRPS().gradient(y=y, mu_pred=mu_test, var_pred=var_test)))


######################################################################################################

y = np.random.uniform(size=[1])

mu = np.random.uniform(size=[1])

variance = np.random.uniform(size=[1])

result = scipy.integrate.quad(lambda x: (float(y <= x) - scipy.stats.norm.cdf(x=(x-mu)/np.sqrt(variance))) ** 2, -np.inf, +np.inf)

print(result)

print(MinusCRPS(min_var=0).evaluate(y=y, mu_pred=mu, var_pred=variance))
print(get_crps(y=y, mu_pred=mu, var_pred=variance))

###################################################

y = np.random.uniform(size=[10])

mu = np.random.uniform(size=[10])

variance = np.random.uniform(size=[10])
print(((y - mu) ** 2).mean())
print(Mse(min_var=0).evaluate(y=y, mu_pred=mu, var_pred=variance))
print(get_mse(y=y, mu_pred=mu))

###################################################

y = np.random.uniform(size=[10])

mu = np.random.uniform(size=[10])

variance = np.random.uniform(size=[10]) + 0.001
print(scipy.stats.norm.logpdf(x=y, loc=mu, scale=np.sqrt(variance)).mean())
print(-LogPredDensity(min_var=0).evaluate(y=y, mu_pred=mu, var_pred=variance))
print(get_gaussian_log_lik(y_test=y, post_mean=mu, post_var=variance))
