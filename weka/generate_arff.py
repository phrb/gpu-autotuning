#! /usr/bin/python

import os
import json
import arff

FLAG_NAMES = [
        u'nvcc:--no-align-double',
        u'nvcc:--use_fast_math',
        u'nvlink:--preserve-relocs',
        u'nvcc:--relocatable-device-code=',
        u'nvcc:--ftz=',
        u'nvcc:--prec-div=',
        u'nvcc:--prec-sqrt=',
        u'ptxas:--fmad=',
        u'ptxas:--allow-expensive-optimizations=',
        u'nvcc:--gpu-architecture=',
        u'ptxas:--def-load-cache=',
        u'ptxas:--opt-level=',
        u'ptxas:--maxrregcount='
]

FLAG_DATA = {
    u'description' : u'',
    u'relation'    : u'Flag Data',
    u'attributes'  : [
        (u'no-align-double',               [u'on', u'off']),
        (u'use_fast_math',                 [u'on', u'off']),
        (u'preserve-relocs',               [u'on', u'off']),
        (u'relocatable-device-code',       [u'true', u'false']),
        (u'ftz',                           [u'true', u'false']),
        (u'prec-div',                      [u'true', u'false']),
        (u'prec-sqrt',                     [u'true', u'false']),
        (u'fmad',                          [u'true', u'false']),
        (u'allow-expensive-optimizations', [u'true', u'false']),
        (u'gpu-architecture',              [u'sm_20', u'sm_21', u'sm_30', u'sm_32', u'sm_35']),
        (u'def-load-cache',                [u'ca', u'cg', u'cv', u'cs']),
        (u'opt-level',                     [u'0', u'1', u'2', u'3']),
        (u'maxrregcount',                  u'NUMERIC')
    ],
    u'data' : [
    ]
}

def build_arff(filename):
    global FLAG_NAMES, FLAG_DATA

    file = open(filename)
    json_file = json.loads(file.read())

    new_data = []
    for flag in FLAG_NAMES:
        new_data.append(json_file[flag])

    FLAG_DATA[u'data'].append(new_data)

    file.close()

def build_for_GPU(gpu_list):
    global FLAG_DATA

    FLAG_DATA[u'data'] = []
    for gpu_name in gpu_list:
        build_arff("../experiments/" + gpu_name + "/MatMulGPU/size_8192_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/MatMulShared/size_8192_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/MatMulSharedUn/size_8192_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/MatMulUn/size_8192_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/SubSeqMax/size_1073741824_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/Bitonic/size_4194304_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/Quicksort/size_65536_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/VecAdd/size_4194304_time_3600/run_0/final_config.json")

    file = open("arff/" + "_".join(gpu_list) + ".arff", "w")
    file.write(arff.dumps(FLAG_DATA))
    file.close()

def build_rodinia_for_GPU(gpu_list):
    global FLAG_DATA

    FLAG_DATA[u'data'] = []
    for gpu_name in gpu_list:
        build_arff("../experiments/" + gpu_name + "/b+tree/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/backprop/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/bfs/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/gaussian/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/heartwall/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/hotspot/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/lavaMD/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/lud/size_default_time_3600/run_0/final_config.json")
        build_arff("../experiments/" + gpu_name + "/myocyte/size_default_time_3600/run_0/final_config.json")

    file = open("arff/Rodinia_" + "_".join(gpu_list) + ".arff", "w")
    file.write(arff.dumps(FLAG_DATA))
    file.close()
    return

def build_rodinia_for(exp_list, gpu_list):
    global FLAG_DATA

    FLAG_DATA[u'data'] = []
    for exp_name in exp_list:
        for gpu_name in gpu_list:
            build_arff("../experiments/" + gpu_name + "/" + exp_name  + "/final_config.json")

    exps = [exp.split("/")[0] for exp in exp_list]
    file = open("arff/Rodinia_" + "_".join(exps) + "_" + "_".join(gpu_list) + ".arff", "w")
    file.write(arff.dumps(FLAG_DATA))
    file.close()
    return


build_for_GPU(["GTX-980"])
build_for_GPU(["Tesla-K40"])
build_for_GPU(["GTX-750"])
build_for_GPU(["GTX-980", "GTX-750", "Tesla-K40"])

build_rodinia_for_GPU(["GTX-980"])
build_rodinia_for_GPU(["Tesla-K40"])
build_rodinia_for_GPU(["GTX-750"])
build_rodinia_for_GPU(["GTX-980", "GTX-750", "Tesla-K40"])

experiments = [
    "b+tree/size_default_time_3600/run_0",
    "backprop/size_default_time_3600/run_0",
    "bfs/size_default_time_3600/run_0",
    "gaussian/size_default_time_3600/run_0",
    "heartwall/size_default_time_3600/run_0",
    "hotspot/size_default_time_3600/run_0",
    "lavaMD/size_default_time_3600/run_0",
    "lud/size_default_time_3600/run_0",
    "myocyte/size_default_time_3600/run_0",
]

gpus = ["GTX-980", "GTX-750", "Tesla-K40"]

build_rodinia_for([experiments[1]], gpus)

build_rodinia_for([experiments[1]], ["GTX-980", "GTX-750"])
build_rodinia_for([experiments[1]], ["Tesla-K40", "GTX-750"])
build_rodinia_for([experiments[1]], ["GTX-980", "Tesla-K40"])
build_rodinia_for([experiments[1]], ["Tesla-K40"])
build_rodinia_for([experiments[1]], ["GTX-750"])
build_rodinia_for([experiments[1]], ["GTX-980"])

##
##init_fail_flags()
##
##parse_fail_flags("../experiments/Tesla-K20/MatMulGPU/size_1024_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/Tesla-K40/MatMulGPU/size_1024_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/GTX-680/MatMulGPU/size_512_time_3600/run_0/failed_configurations.txt")
##
##parse_fail_flags("../experiments/GTX-680/MatMulShared/size_512_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/Tesla-K20/MatMulShared/size_1024_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/Tesla-K40/MatMulShared/size_1024_time_3600/run_0/failed_configurations.txt")
##
##parse_fail_flags("../experiments/GTX-680/MatMulUn/size_128_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/Tesla-K20/MatMulUn/size_1024_time_3600/run_0/failed_configurations.txt")
##parse_fail_flags("../experiments/Tesla-K40/MatMulUn/size_1024_time_3600/run_0/failed_configurations.txt")
##
##parse_fail_flags("../experiments/Tesla-K40/MatMulSharedUn/size_256_time_3600/run_0/failed_configurations.txt")
##
##fig = plt.figure(1, figsize=(10, 7))
##
##ax = fig.add_subplot(111)
##ax.bar(range(len(FFLAGS.keys())), [value / float(TOTAL_FAILED) for value in FFLAGS.values()], 1, color='black')
##
###
### Plot config:
###
##
##ax.set_xticks(np.arange(0.5, len(FFLAGS.keys()), 1))
##ax.set_xticklabels(FFLAGS.keys(), rotation = 90)
##
##ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
##                      alpha=0.5)
##
###ax.set_title("")
##ax.set_ylabel("Percentage of Flags in Failed Configurations")
##
##plt.autoscale()
##fig.tight_layout()
##
##fig.savefig('fail_flags.eps', format = 'eps', dpi = 1000)
