import sys
import numpy as np
from gpy_wrapper.gpy.tests.compute_validation_metrics import launch_full_optimization_evaluations, \
    launch_full_perf_evaluations
import importlib.util
from gpy_wrapper.gpy.libs.transformations import Logexp
import gpy_wrapper.gpy.libs.utils.gpy_estimation_lib as gpy_estimation_lib
import gpy_wrapper.gpy.libs.models.CustomGPy as CustomGPy

##########################################################################

arg_names = ['script_name', 'benchmark_type', 'method_index', 'input_data', 'output_data', 'optional_information']

##########################################################################

args = dict(zip(arg_names, sys.argv))

benchmark_type = args['benchmark_type']

try:
    method_index = int(args['method_index'])
except ValueError:
    raise ValueError("Arg 3 method_index must be an int.")

input_data = args['input_data']
output_data = args['output_data']

if benchmark_type == 'simple':
    benchmark = launch_full_optimization_evaluations
elif benchmark_type == 'loo':
    benchmark = launch_full_perf_evaluations
else:
    raise ValueError("Unknown benchmark : {}".format(benchmark_type))

model = CustomGPy.CustomGPy

if method_index > 9999:
    raise ValueError("Method number if too high.")
else:
    arg_file_name = 'gpy' + '_mle' + str(method_index).zfill(4)
    arg_file_path = 'exps_config' + '.' + 'methods' + '.' + arg_file_name


results_dir_name = arg_file_name

if 'optional_information' in args.keys():
    results_dir_name = results_dir_name.replace('_', '_' + args['optional_information'] + '_')

importlib.util.spec_from_file_location(arg_file_path)

np.random.seed(0)

method_args = importlib.import_module(arg_file_path).method_args

if method_args['param'] == 'log':
    pass
elif method_args['param'] == 'invsoftplus':
    method_args['lengthscale_constraint_class'] = Logexp
    method_args['variance_constraint_class'] = Logexp
else:
    raise ValueError("Unknown parametrization : {}".format(method_args['param']))

del method_args['param']

method_args['profiler'] = gpy_estimation_lib.analytical_mean_and_variance_optimization

method_args['postvar_options'] = {"value": 0, "type": 'truncate'}

if 'input_transform_type' not in method_args.keys():
    method_args['input_transform_type'] = 'None'
if 'output_transform_type' not in method_args.keys():
    method_args['output_transform_type'] = 'None'

benchmark(input_data, output_data, [[results_dir_name, model(**method_args)]])
