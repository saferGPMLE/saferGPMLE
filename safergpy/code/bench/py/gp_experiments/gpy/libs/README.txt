Here are some informations about the arguments. 

I) init :
- classic means : 1) m is initialised with mean(Y)
		  2) sigma^2 is initialised with var(Y)
		  3) ls_i is initialised with std(X_i) for each i

- classic_profiled means : 1) ls_i is initialised with std(X_i) for each i
                           2) m and sigma2 are optimized analytically

- scaled_isotropic_init : We test on a grid of isotropic lengthscales. The possible values are multiples of the X_i's domain diameter. For each lengthscale value we compute the best possible value by optimizing the mean and the variance analytically. For the starting point, we then choose the value that provide the best optimized value. 

- scaled_anisotropic_init : Same as scaled_isotropic_init except we test on a grid of anisotropic lengthscales. The grid is composed of multiples of the X_i's ranges. Note that scaled_isotropic_init and scaled_anisotropic_init are equivalent is the ranges of the X_i's are the same.

- brutal : big random multi start which size is controlled by num_restarts

II) param
For the link function. GPy doesn't work with parameters but with transformed versions of the parameters. Standards kernels are written using the inverse of the softplus function for the variance and the length scales. The softplus is the function : x-> ln(exp(x) + 1) (it's a kind of smooth relu). It means that the length scales and the variance we're looking at are the softplus of what's GPy internally deal with. We tried also to use the log link. The two link functions now availables are :
- log
- invsoftplus for the inverse of the softplus function

III) stopping_criterion
It refers to the stopping criterions of the scipy optimizer. You can either choose strict or soft.

IV) analytical_mu_and_sigma2_optimization
It means that we reestimate the mean and the variance analytically during the process, typically between two restarts.

V) end_analytical_mu_and_sigma2_optimization
Same as previous but only at the very end of the process. This option cannot deteriorate the result so it's on by default.

Note : having analytical_mu_and_sigma2_optimization = False and end_analytical_mu_and_sigma2_optimization = True is the only case where restarts can theoretically deteriorate the value found.

V) do_restarts, restart_sessions_limit, num_restarts

do_restarts = True means that we are launching some random restarts. We don't use GPy restarting method since it doesn't allow to restart from a value which depend on the current point. So we created a dedicated restarting method.
num_restarts : number of restarts per iteration
restart_sessions_limit : number of restarts sessions (a session performs num_restarts retarts).
