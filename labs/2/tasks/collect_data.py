#!/usr/bin/python3

import getopt, sys
import re
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt


program_name = './Merge_sort'
arguments = '1000'
time_token = 'time ='

data = [] # [(threads, mean), ...]
for threads in range (1, 11):
    print("Threads:", threads)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    times = []
    for i in range(0, 8): 
        output = subprocess.run([program_name] + arguments.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        result = re.search('(?<=' + time_token + ').\S*', output).group(0).strip()
        print(result)
        times.append(float(result))
    times.remove(max(times))
    times.remove(min(times))
    mean = sum(times)/5
    print('Mean:', mean, '\n')
    data.append((threads, mean))

df = pd.DataFrame(data, columns = ['Threads', 'Mean Execution Time'])
filename = program_name.split('/')[1] 
df.to_csv(filename + '.csv', index = False)
df.plot(x = 'Threads', y='Mean Execution Time', kind = 'scatter')
plt.title(filename + ' Mean Execution Times')
plt.savefig(filename + '.png')
plt.show()    

# Save so we can leave our machine
