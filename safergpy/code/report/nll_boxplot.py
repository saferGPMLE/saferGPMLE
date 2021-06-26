import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import sys


# --- README ---

'''
This script generates boxplots for NLL differences of default & healed 
as obtained with LOO
'''
# TODO: not clean! too much copy-pasting, to be fixed

# --- To Run ---

'''
Syntax :

python nll_boxplot.py bench_num scheme1 scheme2 dataset_name dimension

Example :

python3 nll_boxplot.py 2 gpy_mle0133 gpy_mle3021 g10 3d
'''


# --- Methods ---

bench_num = sys.argv[1]
method = sys.argv[2]
method1 = sys.argv[3]
dataset =  [str(sys.argv[4]), sys.argv[5]]
print('generating box plots for : \n', [method, method1])

# --- File name parsing utilities ---


def get_problem_and_dimension(file):
    splited_file_name = file.split('_')

    problem = "_".join(file.split('_')[0:(len(splited_file_name) - 1)])
    d = splited_file_name[len(splited_file_name) - 1].replace('.csv', '')

    return problem, d

# --- Let's do the job ---

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench2', 'data_no_std', str(method1))

data_dir_full = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench1', 'proposed', str(method1))

data_dir_healed = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench2', 'data_no_std', str(method))

data_dir_full_healed = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench1', 'proposed', str(method))


# -- Retrieve data from methods ---

cost_dict = {}
cost_dict_healed = {}
cost_dict_full = {}
cost_dict_full_healed = {}

for file in os.listdir(data_dir):
    problem, d = get_problem_and_dimension(file)
    
    if [problem, d] == dataset:

        data = pd.read_csv(os.path.join(data_dir, file), sep=',', index_col=0)[['cost', 'output']]
        df = pd.DataFrame(data)
        
        for output in list(df['output']):
            cost_dict[output] = list(df.loc[df['output'] == output]['cost'])

for file in os.listdir(data_dir_healed):
    problem, d = get_problem_and_dimension(file)
    
    if [problem, d] == dataset:

        data = pd.read_csv(os.path.join(data_dir_healed, file), sep=',', index_col=0)[['cost', 'output']]
        df = pd.DataFrame(data)
        
        for output in list(df['output']):
            cost_dict_healed[output] = list(df.loc[df['output'] == output]['cost'])


for file in os.listdir(data_dir_full):
    problem, d = get_problem_and_dimension(file)
    
    if [problem, d] == dataset:

        data_full = pd.read_csv(os.path.join(data_dir_full, file), sep=',', index_col=0)[['cost', 'output']]
        df_full = pd.DataFrame(data_full)

        for output in list(df_full['output']):
            cost_dict_full[output] = list(df_full.loc[df_full['output'] == output]['cost'])

for file in os.listdir(data_dir_full_healed):
    problem, d = get_problem_and_dimension(file)
    
    if [problem, d] == dataset:

        data_full = pd.read_csv(os.path.join(data_dir_full_healed, file), sep=',', index_col=0)[['cost', 'output']]
        df_full = pd.DataFrame(data_full)

        for output in list(df_full['output']):
            cost_dict_full_healed[output] = list(df_full.loc[df_full['output'] == output]['cost'])

#print('\ncost_dict', cost_dict)
#print('\ncost_dict_healed', cost_dict_healed)
#print('\ncost_dict_full', cost_dict_full)
#print('\ncost_dict_full_healed', cost_dict_full_healed)
#print('\ndiff', np.array(cost_dict['f_1']) - np.array(cost_dict_healed['f_1']))
## Box plot ##

fig = plt.figure(1, figsize=(9, 6))

ax = fig.add_subplot(111)

to_plot = []
to_plot_full = []
for i in list(df_full['output']):
    # print('\n\n{}'.format(i))
    temp = np.array(cost_dict[i]) - np.array(cost_dict_healed[i])
    temp = temp[~np.isnan(temp)]
    to_plot.append(temp)
    to_plot_full.append(np.array(cost_dict_full[i]) - np.array(cost_dict_full_healed[i]))

plot_type = input('Type "h" for histogram : \n')

if plot_type == 'h':

    plt.hist(to_plot, density=False, bins=10, edgecolor='black')
    plt.xlabel('NLL_default - NLL_healed', fontsize=14)
    plt.ylabel('frequency', fontsize=14)
    plt.title('Histogram of NLL differences', fontsize=14)
    plt.show()

else:
        
    bp = ax.boxplot(to_plot)
    bp1  = ax.boxplot(to_plot_full)

    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='g', linewidth=2)
     
    # change outline color, fill color and linewidth of the boxes
    for box in bp1['boxes']:
        # change outline color
        box.set(color='r', linewidth=2)

    # change color and linewidth of the whiskers
    for whisker in bp1['whiskers']:
        whisker.set(color='r', linewidth=2)

    # change color and linewidth of the caps
    for cap in bp1['caps']:
        cap.set(color='r', linewidth=2)

    for median in bp1['medians']:
        median.set(color='r', linewidth=2)

    plt.xlabel('output functions', fontsize=14)
    plt.ylabel('NLL_default - NLL_healed', fontsize=14)
    plt.title('Boxplot of difference in LOO estimated NLL of {}'.format(dataset[0] + '_' + str(dataset[1])), fontsize=14)
    plt.grid(True)
    ax.set_xticklabels(list(df_full['output']))
    plt.show()
