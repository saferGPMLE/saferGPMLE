import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import argparse
import sys

#--- To run ---
'''
Syntax  : python restart_area.py <method_name>
Example : python restart_area.py gpy_mle4021
'''


#--- Argument parser ---
i = 1
method_list = []
while True:
    try:
        method_list.append(sys.argv[i])
        i += 1
    except IndexError:
        break

#print('\nplotting restart area of : ', method_list)

#--- Plot parameters ---

left = -5
right = 400

bottom = 0.25
top = 1.1

n = 2000

if len(method_list) > 1:
    dydx_dict = {}
    area_all = {}
    time_all = {}

for method in method_list:
    
    print('plotting for method : ', method)
    #--- Reading the restart metrics ---

    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'results', 'bench1', 'data_restarts_no_std', method)


    df = pd.DataFrame()
    files = os.listdir(data_dir)
    files.sort()
    for file in files:
        file_path = os.path.join(data_dir, file)
        df_new = pd.read_csv(file_path)

        df_new['file'] = file

        df = df.append(df_new, ignore_index = True)

    ##--- Retrieve best known ---

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
    ##--- Creating bins for the ECDF ---

    bins = np.linspace(left,right,n)

    df_bins = pd.DataFrame(index = bins, columns = batch_nll)

    for b in batch_nll:
        for log_lik_diff in bins:
            prop = ((df[b] - df['best']) < log_lik_diff).mean()
            df_bins.loc[log_lik_diff][b] = prop

    ##--- Computing the area ---
      
    area_dict = {}  
    for b in batch_nll:  
        area = np.trapz(df_bins[b], df_bins.index) 
        area_dict[int(b.split('_')[-1])] = area
        #print('\nfor batch {} area is {}'.format(b, area))
      
    ##--- Computing the total run-time ---

    time_dict = {}
    for b in batch_time:  
        time = np.sum(df[b])
        time_dict[int(b.split('_')[-1])] = time
        #print('\nfor batch {} area is {}'.format(b, area))
     
    ##--- Plotting the area ---

    keys = list(area_dict.keys())
    values = list(area_dict.values())

    area_all[method] = [keys, values]
    area_values = values
    #print(area_values)
    plt.figure(1)
    plt.plot(keys, values)

    plt.ylabel('area under the ECDF', fontsize = 15)
    plt.xlabel('batches of restart/multistart', fontsize = 15)
    plt.title('improvement over restarts', fontsize = 15)



    ##--- Plotting the first derivative ---


    dydx = np.diff(list(values))/np.diff(list(keys))

    #dydx_dict[method] = [list(keys)[1:], dydx]

    plt.figure(2)
    plt.plot(list(keys)[1:], dydx)

    plt.ylabel('1st derivative', fontsize = 15)
    plt.xlabel('batches of restart/multistart', fontsize = 15)
    plt.title('rate of improvement over restarts', fontsize = 15)


    ##--- Plotting the run-time ---

    keys = list(time_dict.keys())
    values = list(time_dict.values())
    
    time_all[method] = [keys, values]
    
    time_values = values
    plt.figure(3)
    plt.plot(keys, values)

    plt.ylabel('cummulative run-time', fontsize = 15)
    plt.xlabel('batches of restart/multistart', fontsize = 15)
    plt.title('cost over restarts', fontsize = 15)


    #plt.show()

'''
if len(method_list) > 1:
    ##--- plotting all method dydx
    plt.figure(4)    
    #labels = ['soft', 'intermediate', 'strict']
    labels = ['int & default', 'strict & default', 'int & healed', 'strict & healed']
    types = [':', ':', '-', '-']
    color = ['orange', 'g', 'orange', 'g']
    i = 0
    for method in method_list:
        plt.plot(dydx_dict[method][0], dydx_dict[method][1], types[i] , label = labels[i], color = color[i])
        i += 1
    plt.ylabel('1st derivative', fontsize = 15)
    plt.xlabel('batches of restart/multistart', fontsize = 15)
    plt.title('improvement rate over restarts', fontsize = 15)

    plt.legend(fontsize = 15)
    plt.show()

'''
##-- for comparing [n_1, 3.0] with n_1 \in 1(1)20--

if len(method_list) > 1:
    plt.figure()    
    xx = []
    yy = []
    for method in method_list:
        xx.append(time_all[method][1][0])
        yy.append(area_all[method][1][0])
    norm = np.max(yy)
    print('max area : ', norm)
    plt.plot(xx, yy/norm, linestyle = '-', marker='o')

    plt.ylabel('area under ECDF', fontsize = 15)
    plt.xlabel('runtime', fontsize = 15)
    plt.title('improvement over restarts', fontsize = 15)
    
    words = list(range(1,21))
    for i in range(len(words)):
        if i%2 ==0 :
            plt.text(xx[i]+0.1, (yy[i]/norm)-0.0005, str(words[i]))

    plt.show()
'''
##-- for comparing n_2*[1, _] --

plt.figure(4)    
norm = np.max(area_values)
norm = np.array([1])
plt.plot(time_values, area_values/norm, linestyle = '-', marker='o')
print('max_area : ', np.max(area_values))
print(time_values)
plt.ylabel('area under ECDF', fontsize = 15)
plt.xlabel('runtime', fontsize = 15)
plt.title('improvement over batches', fontsize = 15)

words = list(range(1,21))
for i in range(len(words)):
    if i%2 ==0 :
        plt.text(time_values[i]+0.1, (area_values[i]/norm)-0.1, str(words[i]))

plt.show()
'''
