#! /usr/bin/python

import os
import matplotlib as mpl
import json

mpl.use('agg')

import matplotlib.pyplot as plt

plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

font = {'family' : 'serif',
        'size'   : 14}

mpl.rc('font', **font)

file = open("experiments/GTX-680/MatMulShared/size_1024_time_3600/run_0/final_config.json")
json_file = json.loads(file.read())

print json_file, type(json_file)


#
# Navigate directories:
#
#for run in os.listdir(d1_path):
#    with open(d1_path + run + "/results.log") as file:
#        best = file.read().splitlines()
#        d1_data.append(float(best[-1].split(" ")[1]))
#

#
# Plot code goes here
#

#
# Plot config:
#
#ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                      alpha=0.5)
#
#ax.set_title("TSP Solution (85900 Cities) Cost After Tuning for 15 minutes (4 runs)")
#ax.set_ylabel("Solution Cost")
#
#fig.tight_layout()
#
#fig.savefig('flags.eps', format = 'eps', dpi = 1000)
#
