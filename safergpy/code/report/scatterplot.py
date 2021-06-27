import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys

# --- To Run ---

'''
Syntax :

python scatterplot.py bench_num scheme_1 scheme_2 dataset_name dimension

Example :

python3 scatterplot.py 2 gpy_mle0133 gpy_mle3021 g10mod 10d
'''


# --- Methods ---

bench_num = sys.argv[1]
methods = [sys.argv[2], sys.argv[3]]
dataset =  [str(sys.argv[4]), sys.argv[5]]
print('Comparing : \n', methods)

# --- File name parsing utilities ---


def get_problem_and_dimension(file):
    splited_file_name = file.split('_')

    problem = "_".join(file.split('_')[0:(len(splited_file_name) - 1)])
    d = splited_file_name[len(splited_file_name) - 1].replace('.csv', '')

    return problem, d

# --- Let's do the job ---

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench'+str(bench_num), 'data')

df = {}

# -- Retrieve data from methods ---

for method_dir in methods:

    df[method_dir] = pd.DataFrame({"post_mean": [], "y_test": [], "ls_dim_1": [], "ls_dim_2": [], "mse": []})

    optim_type = method_dir
    for file in os.listdir(os.path.join(data_dir, method_dir)):
        problem, d = get_problem_and_dimension(file)
        if [problem, d] == dataset:

            data = pd.read_csv(os.path.join(data_dir, method_dir, file), sep=',', index_col=0)[['post_mean', 'y_test', 'ls_dim_1', 'ls_dim_2', 'mse']]

            df[method_dir] = df[method_dir].append(data, ignore_index=True)

            mse_array = np.array(df[method_dir]['mse'])
            mse_array = np.sqrt(mse_array[~np.isnan(mse_array)])

            # log-scale transformations of the lengthscales

            df[method_dir]['ls_dim_1'] = np.log(df[method_dir]['ls_dim_1'])
            df[method_dir]['ls_dim_2'] = np.log(df[method_dir]['ls_dim_2'])

            print('\nLOO MSE for {} is : {}'.format(method_dir, np.mean(mse_array)))


## Scatter plot ##
methods_to_be_compared = methods

plt.figure(1, figsize=(9, 6))
plt.plot(df[methods_to_be_compared[0]]['ls_dim_1'], df[methods_to_be_compared[0]]['ls_dim_2'], 'o', color='blue', label='healed')
plt.plot(df[methods_to_be_compared[1]]['ls_dim_1'], df[methods_to_be_compared[1]]['ls_dim_2'], 'o', color='red', label='default')
plt.ylabel('lengthscale 2', fontsize=14)
plt.xlabel('lengthscale 1', fontsize=14)
plt.legend(prop={'size': 15})
plt.title('scatterplot of lengthscales (in log-scale)', fontsize=14)


plt.figure(2, figsize=(9, 6))
plt.plot(df[methods_to_be_compared[0]]['y_test'], df[methods_to_be_compared[0]]['post_mean'], 'o')
plt.ylabel('predicted output values', fontsize=14)
plt.xlabel('test output values', fontsize=14)
plt.plot(df[methods_to_be_compared[0]]['y_test'], df[methods_to_be_compared[0]]['y_test'], color='r', linestyle='-')
plt.title('observed vs predicted', fontsize=14)

plt.figure(3, figsize=(9, 6))
plt.plot(df[methods_to_be_compared[1]]['y_test'], df[methods_to_be_compared[1]]['post_mean'], 'o')
plt.ylabel('predicted output value', fontsize=14)
plt.xlabel('test output values', fontsize=14)
plt.plot(df[methods_to_be_compared[1]]['y_test'], df[methods_to_be_compared[1]]['y_test'], color='r', linestyle='-')
plt.title('observed vs predicted', fontsize=14)

plt.show()
