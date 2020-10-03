#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

program_name = './MMScanDNC'
size = 6
time_token = 'DNC:'

mean_times = [] # [[threads, mean], ...]
matrices_list = [10000, 40000, 60000, 100000, 200000, 300000, 500000]
for matrices in matrices_list: 
    print("Threads: 16")
    os.environ['OMP_NUM_THREADS'] = str(16)
    results = []
    for i in range(0, 8): 
        output = subprocess.run([program_name] + (matrices + ' ' + str(size)).split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
        print(result)
        results.append(float(result))
    results.remove(max(results))
    results.remove(min(results))
    mean = sum(results)/len(results)
    print('Mean:', mean, '\n')
    mean_times.append((matrices, mean))

data = pd.DataFrame({
    "Matrices": matrices_list, 
    "Mean Execution Time (s)": [datum[1] for datum in mean_times]
})

filename = program_name.split('/')[1] 
data.to_csv(filename + '.csv', index = False)
data.plot(x='Matrices', y='Mean Execution Time (s)', kind = 'scatter')
#plot_multi(data, figsize=(10, 5))
#plt.xticks(np.arange(8), np.arange(1, 9))
#plt.subplots_adjust(right=0.8)
plt.title(filename + ' Statistics')
plt.savefig(filename + '.png')
plt.show()    

