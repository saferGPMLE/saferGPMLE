import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys

#--- Methods ---

bench_num = sys.argv[1]
methods = [sys.argv[2], sys.argv[3]]

print(methods)

#--- File name parsing utilities ---


def get_problem_and_dimension(file):
    splited_file_name = file.split('_')

    problem = "_".join(file.split('_')[0:(len(splited_file_name) - 1)])
    d = splited_file_name[len(splited_file_name) - 1].replace('.csv', '')

    return problem, d

#--- Let's do the job ---

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench'+str(bench_num), 'data_old')

df = pd.DataFrame({"output":[], "cost":[], "problem": [], "optim_type":[], "d":[]})

##-- Retrieve data from methods ---

for method_dir in methods:
    optim_type = method_dir
    for file in os.listdir(os.path.join(data_dir, method_dir)):
        problem, d = get_problem_and_dimension(file)

        data = pd.read_csv(os.path.join(data_dir, method_dir, file), sep=',', index_col=0)[['output', 'cost']]

        data['problem'] = problem
        data['optim_type'] = optim_type
        data['d'] = d

        df = df.append(data, ignore_index=True)

##-- Post processing ---

df_pivot = pd.pivot_table(df, values=['cost'], columns=["optim_type"],index=['problem', 'output', 'd'])

do_anyway = False
if do_anyway:
    df_pivot = df_pivot.dropna()

if df_pivot['cost'].isnull().any().any():
    
    df_nan = df_pivot[df_pivot['cost'].isnull().any(1)]

    if not df_nan[df_nan.columns[1:]].isnull().all(axis=1).all():
        raise ValueError("Comparison not done on same datasets. First example : \n {}".format(df_nan.iloc[0]))

    df_pivot = df_pivot.dropna()


## Scatter plot ##
methods_to_be_compared = methods
df_pivot['diff'] = df_pivot['cost'][methods_to_be_compared[0]] - df_pivot['cost'][methods_to_be_compared[1]]

#print(df_pivot[df_pivot['diff']!=0])

plt.figure(1)
plt.plot(df_pivot['cost'][methods_to_be_compared[0]], df_pivot['cost'][methods_to_be_compared[1]], 'o')
plt.ylabel(methods_to_be_compared[0])
plt.xlabel(methods_to_be_compared[1])
plt.plot(df_pivot['cost'][methods_to_be_compared[0]], df_pivot['cost'][methods_to_be_compared[0]], color='r', linestyle='-')
plt.title('scatterplot of NLL estimates')
plt.figure(2)
plt.plot(range(0, len(df_pivot['diff'])), df_pivot['diff'], 'o')
plt.ylabel('difference in NLL')
plt.xlabel('dataset-outputs')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('difference in NLL estimates for ' + str(methods_to_be_compared[0]) + ' & ' + str(methods_to_be_compared[1]))
plt.show()
