import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse


# --- Argument parser ---

parser = argparse.ArgumentParser()

parser.add_argument("--solid-lines", type=str, nargs='*',
                    help='solid lines curves')

parser.add_argument("--dashed-lines", type=str, nargs='*',
                    help='dashed lines curves')

args = parser.parse_args()

solid_lines = args.solid_lines
dashed_lines = args.dashed_lines

if not (solid_lines or dashed_lines):
    parser.print_help()
    quit()


# --- Methods ---

methods = []

print("solid_lines: " + str(solid_lines))
if solid_lines:
    methods = solid_lines + methods

print("dashed_lines: " + str(dashed_lines))
if dashed_lines:
    methods = dashed_lines + methods


# --- Plot parameters ---

left = -25
right = 400

bottom = 0.25
top = 1.1

n = 1100

# --- File name parsing utilities ---


def get_problem_and_dimension(file):
    splited_file_name = file.split('_')

    problem = "_".join(file.split('_')[0:(len(splited_file_name) - 1)])
    d = splited_file_name[len(splited_file_name) - 1].replace('.csv', '')

    return problem, d


# --- Let's do the job ---
data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench1', 'data')

df = pd.DataFrame({"output": [], "cost": [], "problem": [], "optim_type": [], "d": []})

# -- Retrieve data from methods ---

for method_dir in methods:
    optim_type = method_dir
    for file in os.listdir(os.path.join(data_dir, method_dir)):
        problem, d = get_problem_and_dimension(file)

        data = pd.read_csv(os.path.join(data_dir, method_dir, file), sep=',', index_col=0)[['output', 'cost']]

        data['problem'] = problem
        data['optim_type'] = optim_type
        data['d'] = d

        df = df.append(data, ignore_index=True)
# -- Retrieve best known ---

data = pd.read_csv(os.path.join(data_dir, 'best_known.csv'), sep=',')[['output', 'minimum cost', 'file']]
data = data.rename(columns={'minimum cost': 'cost'})

data['problem'] = data['file'].apply(lambda x: get_problem_and_dimension(x)[0])
data['optim_type'] = 'best_known'
data['d'] = data['file'].apply(lambda x: get_problem_and_dimension(x)[1])

data = data[['output', 'cost', 'problem', 'optim_type', 'd']]

df = df.append(data, ignore_index=True)
summary = "Summary :\n{} datasets\n{} outputs".format(df.groupby(['problem', 'd']).ngroups,
                                                  df.groupby(['problem', 'd', 'output']).ngroups)
# -- Post processing ---

df_pivot = pd.pivot_table(df, values=['cost'], columns=["optim_type"], index=['problem', 'output', 'd'])

# Turn this flag on for comparing methods with different data-sets
# It will only consider the errors for the common datasets

do_anyway = False
if do_anyway:
    df_pivot = df_pivot.dropna()

if df_pivot['cost'].isnull().any().any():

    df_nan = df_pivot[df_pivot['cost'].isnull().any(1)]

    if not df_nan[df_nan.columns[1:]].isnull().all(axis=1).all():
        raise ValueError("Comparison not done on same datasets. First example : \n {}".format(df_nan.iloc[0]))

    df_pivot = df_pivot.dropna()

cost_best = df_pivot['cost']['best_known']

#######

bins = np.linspace(left, right, n)

methods_to_be_compared = methods

df_bins = pd.DataFrame(index=bins, columns=methods_to_be_compared)

for type in methods_to_be_compared:
    # print(type)
    for log_lik_diff in bins:
        prop = ((df_pivot['cost'][type] - cost_best) < log_lik_diff).mean()
        # print("bin : {}, proportion : {}".format(thresold, prop))
        df_bins.loc[log_lik_diff][type] = prop

########


plt.figure()

color = ['b', 'g', 'r', 'c', 'm']

i = 0
if dashed_lines is not None:
    for type in dashed_lines:
        plt.plot(df_bins.index, df_bins[type], ':', label=type, color=color[i])
        i += 1

i = 0
if solid_lines is not None:
    for type in solid_lines:
        plt.plot(df_bins.index, df_bins[type], '-', label=type, color=color[i])
        i += 1

plt.xlabel('l')
plt.ylabel('Pn(all - all_best < l)')
plt.title('cdf de all - all_best')

plt.ylim(bottom=bottom, top=top)
plt.legend(loc="lower right", prop={'size': 14})
plt.show()
