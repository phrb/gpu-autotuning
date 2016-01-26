#! /usr/bin/python

import os
import matplotlib as mpl
import json

mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex = False)
plt.rc('font', family = 'serif')

font = {'family' : 'serif',
        'size'   : 10}

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

FAIL_FLAGS       = { "no-align-double"                : ["on", "off"],
                     "use_fast_math"                  : ["on", "off"],
                     "preserve-relocs"                : ["on", "off"],
                     "relocatable-device-code="       : [ "true", "false" ],
                     "ftz="                           : [ "true", "false" ],
                     "prec-div="                      : [ "true", "false" ],
                     "prec-sqrt="                     : [ "true", "false" ],
                     "fmad="                          : [ "true", "false" ],
                     "allow-expensive-optimizations=" : [ "true", "false" ],
                     "gpu-architecture="              : [ "sm_20", "sm_21",
                                                          "sm_30", "sm_32", "sm_35" ],
                     "def-load-cache="                : [ "ca", "cg", "cv", "cs" ],
                     "opt-level="                     : [ "0", "1", "2", "3" ] }

FLAGS = {}
FFLAGS = {}
TOTAL_FAILED = 0

def init_flags():
    for flag in NVCC_FLAGS:
        for modifier in NVCC_FLAGS[flag]:
            if flag[-1] != "=":
                FLAGS[flag + "=" + modifier] = 0
            else:
                FLAGS[flag + modifier] = 0

def init_fail_flags():
    for flag in FAIL_FLAGS:
        for modifier in FAIL_FLAGS[flag]:
            if flag[-1] != "=":
                FFLAGS[flag + "=" + modifier] = 0
            else:
                FFLAGS[flag + modifier] = 0

def clear_flags():
    for flag in FLAGS:
        FLAGS[flag] = 0

def clear_fail_flags():
    for flag in FFLAGS:
        FFLAGS[flag] = 0

def parse_json_flags(filename):
        file = open(filename)
        json_file = json.loads(file.read())

        for flag in json_file:
            if flag != "ptxas:--maxrregcount=":
                if flag[-1] != "=":
                    parameter = flag + "=" + json_file[flag]
                else:
                    parameter = flag + json_file[flag]

                if parameter in FLAGS:
                    FLAGS[parameter] += 1

        file.close()

def parse_fail_flags(filename):
    global TOTAL_FAILED
    file = open(filename)
    lines = file.read().split("failed_example_cmd: ")
    for line in lines:
        TOTAL_FAILED += 1
        flags = [flag.split(" ", 1)[0] for flag in line.split("--")[1:]]
        if "no-align-double" in flags:
            flags.remove("no-align-double")
            FFLAGS["no-align-double=on"] += 1
        else:
            FFLAGS["no-align-double=off"] += 1

        if "use_fast_math" in flags:
            flags.remove("use_fast_math")
            FFLAGS["use_fast_math=on"] += 1
        else:
            FFLAGS["use_fast_math=off"] += 1

        if "preserve-relocs" in flags:
            flags.remove("preserve-relocs")
            FFLAGS["preserve-relocs=on"] += 1
        else:
            FFLAGS["preserve-relocs=off"] += 1

        for flag in flags:
            opt, mod = flag.split("=")
            if opt != "maxrregcount":
                parameter = opt + "=" + mod
                if parameter in FFLAGS:
                    FFLAGS[parameter] += 1

init_flags()

# GTX-980 MatMul

parse_json_flags("../experiments/GTX-980/MatMulGPU/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-980/MatMulShared/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-980/MatMulSharedUn/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-980/MatMulUn/size_16384_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("MatMul Autotuned Flags in the GTX-980")

plt.autoscale()
fig.tight_layout()

fig.savefig('matmul_gtx980_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-750 MatMul

parse_json_flags("../experiments/GTX-750/MatMulGPU/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-750/MatMulShared/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-750/MatMulSharedUn/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/GTX-750/MatMulUn/size_16384_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("MatMul Autotuned Flags in the GTX-750")

plt.autoscale()
fig.tight_layout()

fig.savefig('matmul_gtx750_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# Tesla-K40 MatMul

parse_json_flags("../experiments/Tesla-K40/MatMulGPU/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/Tesla-K40/MatMulShared/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/Tesla-K40/MatMulSharedUn/size_16384_time_3600/run_0/final_config.json")
parse_json_flags("../experiments/Tesla-K40/MatMulUn/size_16384_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("MatMul Autotuned Flags in the Tesla-K40")

plt.autoscale()
fig.tight_layout()

fig.savefig('matmul_teslak40_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# Tesla-K40 SubSeqMax

parse_json_flags("../experiments/Tesla-K40/SubSeqMax/size_1073741824_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("SubSeqMax Autotuned Flags in the Tesla-K40")

plt.autoscale()
fig.tight_layout()

fig.savefig('subseq_teslak40_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-980 SubSeqMax

parse_json_flags("../experiments/GTX-980/SubSeqMax/size_1073741824_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("SubSeqMax Autotuned Flags in the GTX-980")

plt.autoscale()
fig.tight_layout()

fig.savefig('subseq_gtx980_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-750 SubSeqMax

parse_json_flags("../experiments/GTX-750/SubSeqMax/size_1073741824_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("SubSeqMax Autotuned Flags in the GTX-750")

plt.autoscale()
fig.tight_layout()

fig.savefig('subseq_gtx750_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-750 Bitonic

parse_json_flags("../experiments/GTX-750/Bitonic/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Bitonic Autotuned Flags in the GTX-750")

plt.autoscale()
fig.tight_layout()

fig.savefig('bitonic_gtx750_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-980 Bitonic

parse_json_flags("../experiments/GTX-980/Bitonic/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Bitonic Autotuned Flags in the GTX-980")

plt.autoscale()
fig.tight_layout()

fig.savefig('bitonic_gtx980_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# Tesla-K40 Bitonic

parse_json_flags("../experiments/Tesla-K40/Bitonic/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Bitonic Autotuned Flags in the Tesla-K40")

plt.autoscale()
fig.tight_layout()

fig.savefig('bitonic_teslak40_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# Tesla-K40 Quicksort

parse_json_flags("../experiments/Tesla-K40/Quicksort/size_65536_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Quicksort Autotuned Flags in the Tesla-K40")

plt.autoscale()
fig.tight_layout()

fig.savefig('quicksort_teslak40_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-750 Quicksort

parse_json_flags("../experiments/GTX-750/Quicksort/size_65536_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Quicksort Autotuned Flags in the GTX-750")

plt.autoscale()
fig.tight_layout()

fig.savefig('quicksort_gtx750_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-980 Quicksort

parse_json_flags("../experiments/GTX-980/Quicksort/size_65536_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("Quicksort Autotuned Flags in the GTX-980")

plt.autoscale()
fig.tight_layout()

fig.savefig('quicksort_gtx980_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-980 VecAdd

parse_json_flags("../experiments/GTX-980/VecAdd/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("VecAdd Autotuned Flags in the GTX-980")

plt.autoscale()
fig.tight_layout()

fig.savefig('vecadd_gtx980_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# GTX-750 VecAdd

parse_json_flags("../experiments/GTX-750/VecAdd/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("VecAdd Autotuned Flags in the GTX-750")

plt.autoscale()
fig.tight_layout()

fig.savefig('vecadd_gtx750_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

# Tesla-K40 VecAdd

parse_json_flags("../experiments/Tesla-K40/VecAdd/size_4194304_time_3600/run_0/final_config.json")

fig = plt.figure(1, figsize=(10, 7))

ax = fig.add_subplot(111)
ax.bar(range(len(FLAGS.keys())), FLAGS.values(), 1, color='black')

#
# Plot config:
#

ax.set_xticks(np.arange(0.5, len(FLAGS.keys()), 1))
ax.set_xticklabels([key.split(":--")[1] for key in FLAGS.keys()], rotation = 90)

ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                      alpha=0.5)

#ax.set_title("")
ax.set_ylabel("VecAdd Autotuned Flags in the Tesla-K40")

plt.autoscale()
fig.tight_layout()

fig.savefig('vecadd_teslak40_flags.eps', format = 'eps', dpi = 1000)

plt.clf()

clear_flags()

#
#init_fail_flags()
#
#parse_fail_flags("../experiments/Tesla-K20/MatMulGPU/size_1024_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/Tesla-K40/MatMulGPU/size_1024_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/GTX-680/MatMulGPU/size_512_time_3600/run_0/failed_configurations.txt")
#
#parse_fail_flags("../experiments/GTX-680/MatMulShared/size_512_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/Tesla-K20/MatMulShared/size_1024_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/Tesla-K40/MatMulShared/size_1024_time_3600/run_0/failed_configurations.txt")
#
#parse_fail_flags("../experiments/GTX-680/MatMulUn/size_128_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/Tesla-K20/MatMulUn/size_1024_time_3600/run_0/failed_configurations.txt")
#parse_fail_flags("../experiments/Tesla-K40/MatMulUn/size_1024_time_3600/run_0/failed_configurations.txt")
#
#parse_fail_flags("../experiments/Tesla-K40/MatMulSharedUn/size_256_time_3600/run_0/failed_configurations.txt")
#
#fig = plt.figure(1, figsize=(10, 7))
#
#ax = fig.add_subplot(111)
#ax.bar(range(len(FFLAGS.keys())), [value / float(TOTAL_FAILED) for value in FFLAGS.values()], 1, color='black')
#
##
## Plot config:
##
#
#ax.set_xticks(np.arange(0.5, len(FFLAGS.keys()), 1))
#ax.set_xticklabels(FFLAGS.keys(), rotation = 90)
#
#ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
#                      alpha=0.5)
#
##ax.set_title("")
#ax.set_ylabel("Percentage of Flags in Failed Configurations")
#
#plt.autoscale()
#fig.tight_layout()
#
#fig.savefig('fail_flags.eps', format = 'eps', dpi = 1000)
