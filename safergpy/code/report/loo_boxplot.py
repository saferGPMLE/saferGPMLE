import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import sys

# --- README ---

'''
This script generates boxplots for range(lengthscale) values for LOO
'''

# --- To Run ---

'''
Syntax :

python loo_boxplot.py bench_num scheme dataset_name dimension output

Example :

python3 loo_boxplot.py 2 gpy_mle0133 g10 3d f_1 
'''


# --- Methods ---

bench_num = sys.argv[1]
method = sys.argv[2]
dataset =  [str(sys.argv[3]), sys.argv[4]]
output = str(sys.argv[5])
print('generating box plots for : \n', method)

# --- File name parsing utilities ---


def get_problem_and_dimension(file):
    splited_file_name = file.split('_')

    problem = "_".join(file.split('_')[0:(len(splited_file_name) - 1)])
    d = splited_file_name[len(splited_file_name) - 1].replace('.csv', '')

    return problem, d

# --- Let's do the job ---

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench2', 'data', str(method))

data_dir_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench1', 'data', str(method))


# -- Retrieve data from methods ---

ls_dict = {}

for file in os.listdir(data_dir):
    problem, d = get_problem_and_dimension(file)

    if [problem, d] == dataset:

        data = pd.read_csv(os.path.join(data_dir, file), sep=',', index_col=0)
        df = pd.DataFrame(data)

        ls = []
        for i in df.columns:
            if 'ls_dim' in i:
                ls.append(i)

        df = df.loc[df['output'] == output]

        for i in ls:
            df[i] = np.log(df[i])


for file in os.listdir(data_dir_full):
    problem, d = get_problem_and_dimension(file)

    if [problem, d] == dataset:

        data_full = pd.read_csv(os.path.join(data_dir_full, file), sep=',', index_col=0)
        df_full = pd.DataFrame(data_full)

        df_full = df_full.loc[df_full['output'] == output]

        for i in ls:
            df_full[i] = np.log(df_full[i])


## Box plot ##

fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)

boxplot = df.boxplot(column=ls, grid=False, color=dict(boxes='black', whiskers='black', caps='black'))

boxplot2 = df_full.boxplot(column=ls, grid=False, color=dict(boxes='r', whiskers='r', medians='r', caps='r'))

plt.xlabel('input dimensions', fontsize=14)
plt.ylabel('range (in log scale)', fontsize=14)
plt.title('Boxplot of LOO estimated range of {}, {} with {}'.format(dataset[0] + '_' + dataset[1], output, 'improved'), fontsize=14)
labels = [x.split('_')[-1] for x in ls]
ax.set_xticklabels(labels)
# plt.grid(True)
# Major ticks every 20, minor ticks every 5
ax.set_ylim(-5, 30)
major_ticks = np.arange(0, 30, 5)

ax.set_yticks(major_ticks)

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='major', alpha=0.5)
plt.show()
