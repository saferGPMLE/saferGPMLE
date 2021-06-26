import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse
import sys

# --- To run ---
'''
Syntax  : python restart_area.py <method_name>
Example : python restart_area.py gpy_mle4021
'''


# --- Argument parser ---
i = 1
method_list = ['gpy_mle1220', 'gpy_mle2220']


#print('\nplotting restart area of : ', method_list)

# --- Plot parameters ---

left = -5
right = 400

bottom = 0.25
top = 1.1

n = 2000

area_all = {}
time_all = {}

for method in method_list:
    
    print('plotting for method : ', method)
    # --- Reading the restart metrics ---

    data_dir = os.path.join(os.getcwd(), '..', '..', 'results', 'bench1', 'restarts', method)

    df = pd.DataFrame()
    files = os.listdir(data_dir)
    files.sort()
    for file in files:
        file_path = os.path.join(data_dir, file)
        df_new = pd.read_csv(file_path)

        df_new['file'] = file

        df = df.append(df_new, ignore_index=True)

    # --- Retrieve best known ---

    data = pd.read_csv(os.path.join(data_dir, '../', 'best_known.csv'), sep=',')[['minimum cost', 'file']]

    df['best'] = data['minimum cost']

    col = df.columns
    batch_nll = []
    batch_time = []
    for i in col:
        if 'nll' in i:
            batch_nll.append(i)
        elif 'time' in i:
            batch_time.append(i)
    batch_nll.sort(key=lambda x: int(x.split('_')[-1])) 
    batch_time.sort(key=lambda x: int(x.split('_')[-1])) 

    # for debugging:
    '''
    for ind in df.index: 
        print('\nfor df_b : {}, df_best : {},  file is {}'.format(df[batch_nll[-1]][ind], df['best'][ind], df['file'][ind]))
        if df[batch_nll[0]][ind] < df['best'][ind]:
            print('++++++++++++++++')
            print('\n{}{}, b {}, best {}, diff {}'.format(df['problem'][ind], df['d'][ind], df[batch_nll[0]][ind], df['best'][ind],  df[batch_nll[0]][ind] - df['best'][ind]))
            print('================')
    ''' 
    # --- Creating bins for the ECDF ---

    bins = np.linspace(left, right, n)

    df_bins = pd.DataFrame(index=bins, columns=batch_nll)

    for b in batch_nll:
        for log_lik_diff in bins:
            prop = ((df[b] - df['best']) < log_lik_diff).mean()
            df_bins.loc[log_lik_diff][b] = prop

    # --- Computing the area ---
      
    area_dict = {}  
    for b in batch_nll:  
        area = np.trapz(df_bins[b], df_bins.index) 
        area_dict[int(b.split('_')[-1])] = area
        #print('\nfor batch {} area is {}'.format(b, area))
      
    # --- Computing the total run-time ---

    time_dict = {}
    for b in batch_time:  
        time = np.sum(df[b])
        time_dict[int(b.split('_')[-1])] = time
        #print('\nfor batch {} area is {}'.format(b, area))
     
    # --- Plotting the area ---

    keys = list(area_dict.keys())
    values = list(area_dict.values())

    area_values = values
    area_all[method] = area_values

    # --- Plotting the run-time ---

    keys = list(time_dict.keys())
    values = list(time_dict.values())
    
    time_values = values
    time_all[method] = time_values
    
    
# -- for comparing n_2*[1, _] --

fig = plt.figure(1, figsize=(9, 6))

ax = fig.add_subplot(111)

method_name = {'gpy_mle1220': 'strict', 'gpy_mle2220': 'soft'}
 
norm = np.float64(388.1727536396725)
print('max area is {}'.format(np.max(area_all['gpy_mle1220'] + area_all['gpy_mle2220'])))

for method in method_list:
    plt.plot(time_all[method], area_all[method]/norm, linestyle='-', marker='o', label=method_name[method])
    
    words = list(range(1, 21))
    for i in range(len(words)):
        if i%2 ==0:
            plt.text(time_all[method][i]-5, (area_all[method][i]/norm)-0.0005, str(words[i]), fontsize=14)


#print('max_area : ', np.max(area_values))
# print(time_values)
plt.ylabel('area under ECDF', fontsize=20)
plt.xlabel('runtime', fontsize=20)
#plt.title('improvement over restarts', fontsize = 20)
plt.legend(fontsize=20)

ax.set_yticks([0.990, 0.992, 0.994, 0.996, 0.998, 1])

ax.tick_params(axis="x", labelsize=20)    
ax.tick_params(axis="y", labelsize=14)

plt.grid(True)

plt.show()
