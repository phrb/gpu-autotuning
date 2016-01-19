#! /usr/bin/python2

import os
import time
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument( "-cp", "--cuda-path",
                        dest     = "cuda_path",
                        type     = str,
                        required = True,
                        help     = "The path for CUDA libraries.")
argparser.add_argument( "-b", "--baseline-only",
                        dest = "baseline",
                        action = "store_true",
                        default = False,
                        help    = "Runs baseline calculations only.")
argparser.add_argument( "-t", "--tune-only",
                        dest = "tune",
                        action = "store_true",
                        default = False,
                        help    = "Runs tunning only.")

def tune(program, arguments, logdir, run_time, runs, benchmark, cuda_path):
    os.system("mkdir -p " + logdir)
    cmd = "./scripts/run_experiments.py"
    cmd += " -f="        + program
    cmd += " -fargs="    + "\"" + " ".join(arguments) + "\""
    cmd += " -ld="       + logdir
    cmd += " -lc=final"
    cmd += " -time="     + str(run_time)
    cmd += " -th=1"
    cmd += " -r="        + str(runs)
    cmd += " -br="       + str(benchmark)
    cmd += " -cp="       + "\"" + cuda_path + "\""
    os.system(cmd)

def baseline(program, arguments, logdir, runs, cuda_path):
    options = "-Xptxas --opt-level="
    values  = ["0", "1", "2", "3"]

    os.system("mkdir -p " + logdir + "_baseline" )
    for value in values:
        # Compiling:
        print "nvcc -w " + cuda_path + program + " -o tmp.bin " + options + value
        os.system("nvcc -w " + cuda_path + program + " -o tmp.bin " + options + value)
        for i in range(runs):
            cmd   = "./tmp.bin " + " ".join(arguments) + " "
            logfile  = logdir + "_baseline/opt_" + value + ".txt"

            print cmd
            start   = time.time()
            os.system(cmd)
            end     = time.time()

            with open(logfile, "a+") as file:
                file.write(str(end - start) + "\n")

def run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args):
    print "[INFO] Starting " + program + " Experiments."
    os.system("mkdir -p " + logdir )
    for i in steps:
        print "[INFO] Size: " + str(i)
        if args.baseline == False:
            logs = logdir + "/size_" + "'" + str(i) + "'" + "_time_" + str(run_time)
            tune(program, [str(i), arguments], logs, run_time, runs, benchmark, cuda_path)
        if args.tune == False:
            print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
            baseline(program, [str(i), arguments], logdir + "/size_" + str(i), benchmark, cuda_path)
            print "[INFO] Baseline Calculation Done."
    print "[INFO] " + program + " Experiments Done."

#
# Common:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
run_time    = 60
runs        = 1
benchmark   = 1

#
# MatMulShared Experiments:
#
#program     = "../matMul/matMul_gpu_sharedmem.cu"
#logdir      = "logs/MatMulShared"
#arguments   = "16 0"
#steps       = [2048, 4096, 8192, 16384, 32768]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
##
## TODO: Fix MatMulSharedUn for all sizes.
##
## MatMulSharedUn Experiments:
##
#steps       = [2048, 4096, 8192, 16384, 32768]
#program     = "../matMul/matMul_gpu_sharedmem_uncoalesced.cu"
#logdir      = "logs/MatMulSharedUn"
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
##
## MatMulUn Experiments:
##
#steps       = [2048, 4096, 8192, 16384, 32768]
#program     = "../matMul/matMul_gpu_uncoalesced.cu"
#logdir      = "logs/MatMulUn"
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
##
## MatMulGPU Experiments:
##
#steps       = [2048, 4096, 8192, 16384, 32768]
#program     = "../matMul/matMul_gpu.cu"
#logdir      = "logs/MatMulGPU"
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
##
## SubSeqMax Experiments:
##
#program     = "../bioinformatic/SubSeqMax.cu"
#logdir      = "logs/SubSeqMax"
#arguments   = "0"
#steps       = [2**26, 2**27, 2**28, 2**29, 2**30]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
##
## Bitonic Sort Experiments:
##
#program     = "../sorting/bitonic_sort.cu"
#logdir      = "logs/Bitonic"
#arguments   = "0"
#steps       = [2**18, 2**19, 2**20, 2**21, 2**22]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
##
## Quicksort Experiments:
##
#program     = "../sorting/quicksort.cu"
#logdir      = "logs/Quicksort"
#arguments   = "0"
#steps       = [2**12, 2**13, 2**14, 2**15, 2**16]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
##
## Vector Addition Experiments:
##
#program     = "../vectorAdd/vectorAdd.cu"
#logdir      = "logs/VecAdd"
#arguments   = " "
#steps       = [2**15, 2**17, 2**18, 2**20, 2**22]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)

## Applications do Benchmark Rodinia:
#
##
## Rodinia: Particle Filter:
##
#program     = "../rodinia_3.0/cuda/particlefilter/ex_particle_CUDA_naive_seq.cu"
#logdir      = "logs/ParticleFilterNaive"
#arguments   = "-x 128 -y 128 -z 10 -np "
#steps       = [1000, 5000, 10000, 20000, 50000]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
##
## Rodinia: Particle Filter:
##
#program     = "../rodinia_3.0/cuda/particlefilter/ex_particle_CUDA_float_seq.cu"
#logdir      = "logs/ParticleFilterFloat"
#arguments   = "-x 128 -y 128 -z 10 -np "
#steps       = [1000, 5000, 10000, 20000, 50000]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
##
## Rodinia: Pathfinder:
##
#program     = "../rodinia_3.0/cuda/pathfinder/pathfinder.cu"
#logdir      = "logs/Pathfinder"
#arguments   = " "
#steps       = ["100000", "1000000", "5000000", "10000000"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
##
## Rodinia: Hotspot:
##
#program     = "../rodinia_3.0/cuda/hotspot/hotspot.cu"
#logdir      = "logs/Hotspot"
#steps       = ["512 2 2 ../../../rodinia_3.0/data/hotspot/temp_512  ../../../rodinia_3.0/data/hotspot/power_512 output.out"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
##
## Rodinia: Gaussian limination:
##
#program     = "../rodinia_3.0/cuda/gaussian/gaussian.cu"
#logdir      = "logs/Gaussian"
#arguments   = "-q -s "
#steps       = [128, 256, 512, 1024, 2048]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)

#
# TODO: Write code for the other experiments.
#
#
# Rodinia: Gaussian limination:
#
program     = "../rodinia_3.0/cuda/backprop"
logdir      = "logs/backprop"
arguments   = " "
steps       = ["default"]
#
run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
