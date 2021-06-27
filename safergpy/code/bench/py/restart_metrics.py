import sys
import os

# TODO(): make it an argument ?
os.environ["OMP_NUM_THREADS"] = str(1)

import numpy as np
import pandas as pd
from gp_experiments.gpy.tests.compute_validation_metrics import launch_full_optimization_evaluations, \
    launch_full_perf_evaluations
import importlib.util
from gp_experiments.gpy.libs.transformations import Logexp
import gp_experiments.gpy.libs.utils.gpy_estimation_lib as gpy_estimation_lib

# --- To run ---
'''
Syntax :  python3 code/bench/py/restart_metrics.py simple gpy <method_num> <input_dir> <output_dir>
Exampe :  python3 code/bench/py/restart_metrics.py simple gpy 5021 datasets results/bench1/restart
'''

##########################################################################

toolbox_index = {'gpy': '0'}
arg_names = ['script_name', 'benchmark_type', 'toolbox', 'method_index', 'input_data', 'output_data', 'optional_information']

##########################################################################

args = dict(zip(arg_names, sys.argv))

benchmark_type = args['benchmark_type']
toolbox = args['toolbox']

try:
    method_index = int(args['method_index'])
except ValueError:
    raise ValueError("Arg 3 method_index must be an int.")

input_data = args['input_data']
output_data = args['output_data']

if toolbox == 'gpy':
    import gp_exp_misc.CustomGPyMonitoredRestarts as CustomGPyMonitoredRestarts
    model = CustomGPyMonitoredRestarts.CustomGPyMonitoredRestarts
else:
    raise ValueError("Unknown toolbox : {}".format(toolbox))

# alternate for GPy only setup

if method_index > 9999:
    raise ValueError("Method number if too high.")
else:
    arg_file_name = toolbox + '_mle' + str(method_index).zfill(4)
    arg_file_path = toolbox + '.' + 'restart_methods' + '.' + arg_file_name


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

# Upto this part is just copy-pasted from launcher.py
# TODO : it may contain things not needed, have to clean it.

model_instance = model(**method_args)

target_dir = os.path.join(output_data, results_dir_name)
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for file in os.listdir(input_data):
    print('\n\nWorking on file : ', file)
    data_path = os.path.join(input_data, file)
    data = pd.read_csv(data_path, sep=',', index_col=0)
    predictors = [a for a in data.columns if 'x' in a]
    outputs = [a for a in data.columns if 'x' not in a]
    data['row'] = range(data.shape[0])

    df = pd.DataFrame()
    for output in outputs:
        print('\nWorking on output : ', output)
        x_train = data[predictors].values
        y_train = data[[output]].values

        model_instance.clean()

        model_instance.set_data(x_train, y_train)
        l = model_instance.train()
        l['output'] = output
        df = df.append(l, ignore_index=True)

    df.to_csv(os.path.join(output_data, results_dir_name, file))
