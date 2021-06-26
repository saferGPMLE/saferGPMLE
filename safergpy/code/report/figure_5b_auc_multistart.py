import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()

N_repeatitions = 50
N_multistarts = 20


area_dict = {}
time_dict = {}

args = ['gpy_mle1121', 'gpy_mle1122']

for arg in args:

    data_dir = os.path.join(cwd, '../../results/bench1/multistarts/', arg)

    df = pd.DataFrame()
    files = os.listdir(data_dir)
    files.sort()

    for file in files:
        file_path = os.path.join(data_dir, file)
        df_new = pd.read_csv(file_path, index_col=0)

        df_new['file'] = file

        df = df.append(df_new, ignore_index=True)

    # Retrieving the best known
    data = pd.read_csv(os.path.join(data_dir, '../', 'best_known.csv'), sep=',')[['minimum cost', 'file', 'output']]

    df = pd.merge(df, data, on=['file', 'output'])

    #init_draw = df[['nll_1', 'time_1', 'file', 'output']]

    #df = df[[c for c in df.columns if c not in ['nll_1', 'time_1']]]

    results_times = {}
    results_areas = {}

    for i in range(N_repeatitions):
        print('i : {}'.format(i))

        keys_nll = [x for x in df.columns if ('nll' in x) and (int(x.split('_')[1]) % N_repeatitions == i) and (int(x.split('_')[1]) != 1)]
        keys_nll.sort(key=lambda x: int(x.split('_')[-1]))

        keys_time = [x for x in df.columns if ('time' in x) and (int(x.split('_')[1]) % N_repeatitions == i) and (int(x.split('_')[1]) != 1)]
        keys_time.sort(key=lambda x: int(x.split('_')[-1]))

        assert [int(x.split('_')[1]) for x in keys_nll] == [int(x.split('_')[1]) for x in keys_time]

        key_nll_with_start = ['nll_1'] + keys_nll
        key_time_with_start = ['time_1'] + keys_time

        assert len(key_nll_with_start) == N_multistarts and len(key_time_with_start) == N_multistarts

        nll_minima = np.minimum.accumulate(df[key_nll_with_start].values, axis=1)
        time_sum = np.cumsum(df[key_time_with_start].values, axis=1)

        nll_table = pd.concat((pd.DataFrame(nll_minima, columns=key_nll_with_start), df[['file', 'output', 'minimum cost']]), axis=1)
        time_table = pd.concat((pd.DataFrame(time_sum, columns=key_time_with_start), df[['file', 'output']]), axis=1)

        assert len(key_nll_with_start) == len(key_time_with_start)

        for j in range(len(key_nll_with_start)):
            left = -5
            right = 400
            n = 1100

            bins = np.linspace(left, right, n)

            temp = []
            for log_lik_diff in bins:
                prop = ((nll_table[key_nll_with_start[j]] - nll_table['minimum cost']) < log_lik_diff).mean()
                temp.append(prop)

            props = temp

            time = time_table[key_time_with_start[j]].sum()
            area = np.trapz(props, bins)

            if j not in results_times.keys():
                results_times[j] = [time]
            else:
                results_times[j].append(time)

            if j not in results_areas.keys():
                results_areas[j] = [area]
            else:
                results_areas[j].append(area)

    area_dict[arg] = [results_areas]
    time_dict[arg] = [results_times]


fig = plt.figure(1, figsize=(9, 6))

ax = fig.add_subplot(111)

method_name = {'gpy_mle1122': 'strict', 'gpy_mle1121': 'soft'}

norm = np.max([np.array(area_dict['gpy_mle1121'][0][s]).mean() for s in range(N_multistarts)] +
              [np.array(area_dict['gpy_mle1122'][0][s]).mean() for s in range(N_multistarts)])

print('max area is {}'.format(norm))

for arg in args[::-1]:

    #norm = np.max([np.array(area_dict[arg][0][s]).mean() for s in range(N_multistarts)])
    #print('\nmax area for {} is {}'.format(arg, norm))
    plt.plot(
        [np.array(time_dict[arg][0][s]).mean() for s in range(N_multistarts)],
        [np.array(area_dict[arg][0][s]).mean() for s in range(N_multistarts)]/norm,
        linestyle='-',
        marker='o', label=method_name[arg]
    )

    if arg == 'gpy_mle1122':

        words = list(range(1, 21))
        for i in range(len(words)):
            if i%2 == 0:
                plt.text([np.array(time_dict[arg][0][s]).mean() for s in range(N_multistarts)][i]+0.5,
                         ([np.array(area_dict[arg][0][s]).mean() for s in range(N_multistarts)][i]/norm)-0.0005,
                         str(words[i]), fontsize=14)

    else:

        words = list(range(1, 21))
        for i in range(len(words)):
            if i%2 == 0:
                plt.text([np.array(time_dict[arg][0][s]).mean() for s in range(N_multistarts)][i]-30,
                         ([np.array(area_dict[arg][0][s]).mean() for s in range(N_multistarts)][i]/norm)+0.00015,
                         str(words[i]), fontsize=14)


plt.ylabel('area under ECDF', fontsize=20)
plt.xlabel('runtime', fontsize=20)
#plt.title('improvement over multistarts', fontsize = 20)
plt.legend(fontsize=20)

ax.set_yticks([0.990, 0.992, 0.994, 0.996, 0.998, 1])

ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=14)

plt.grid(True)

plt.show()
