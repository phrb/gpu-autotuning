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


NVCC_FLAGS       = { "nvcc:--no-align-double"                 : ["on", "off"],
                     "nvcc:--use_fast_math"                   : ["on", "off"],
                     "nvlink:--preserve-relocs"               : ["on", "off"],
                     "nvcc:--relocatable-device-code="        : [ "true", "false" ],
                     "nvcc:--ftz="                            : [ "true", "false" ],
                     "nvcc:--prec-div="                       : [ "true", "false" ],
                     "nvcc:--prec-sqrt="                      : [ "true", "false" ],
                     "ptxas:--fmad="                          : [ "true", "false" ],
                     "ptxas:--allow-expensive-optimizations=" : [ "true", "false" ],
                     "nvcc:--gpu-architecture="               : [ "sm_20", "sm_21",
                                                                  "sm_30", "sm_32", "sm_35" ],
                     "ptxas:--def-load-cache="                : [ "ca", "cg", "cv", "cs" ],
                     "ptxas:--opt-level="                     : [ "0", "1", "2", "3" ] }
PTXAS_NUM_PARAMS = [ ( "ptxas:--maxrregcount=", 16, 63 ) ]

FLAGS = {}

for flag in NVCC_FLAGS:
    for modifier in NVCC_FLAGS[flag]:
        FLAGS[flag + modifier] = 0

print FLAGS

file = open("experiments/GTX-680/MatMulShared/size_1024_time_3600/run_0/final_config.json")
json_file = json.loads(file.read())

print json_file, type(json_file)

for flag in json_file:
    if flag != "ptxas:--maxrregcount=":
        print flag
        parameter = flag + json_file[flag]
        if parameter in FLAGS:
            FLAGS[parameter] += 1

print FLAGS

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
