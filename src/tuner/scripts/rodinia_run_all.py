#! /usr/bin/python2

import os
import time
import argparse
import subprocess

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
    cmd = "./scripts/rodinia_run_experiments.py"
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
        os.environ["NVCC_FLAGS"] = ""
        old_path = os.getcwd()
        os.chdir(program)
        subprocess.call("make clean", shell = True)
        subprocess.call("rm -f *.o *~ *.linkinfo", shell = True)

        os.environ["NVCC_FLAGS"] = "-w " + cuda_path + " " + options + value + " "
        compile_result = subprocess.call("make",
                                         shell = True)
        os.chdir(old_path)

        for i in range(runs):
            logfile  = logdir + "_baseline/opt_" + value + ".txt"

            old_path = os.getcwd()
            os.chdir(program)
            start   = time.time()
            subprocess.call("./run", shell = True)
            end     = time.time()
            os.chdir(old_path)

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
run_time    = 7200
runs        = 1
benchmark   = 8

#
# Rodinia: Back Propagation 
#
program     = "../rodinia_3.0/cuda/backprop"
logdir      = "logs/backprop"
arguments   = " "
steps       = ["default"]
#
run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)

#
# Rodinia: Breadth-First Search
#
program     = "../rodinia_3.0/cuda/bfs"
logdir      = "logs/bfs"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)

#
# Rodinia: B+ Tree
#
program     = "../rodinia_3.0/cuda/b+tree"
logdir      = "logs/b+tree"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)


#
# Rodinia: Gaussian Elimination
#
program     = "../rodinia_3.0/cuda/gaussian"
logdir      = "logs/gaussian"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)


#
# Rodinia: Heartwall
#
program     = "../rodinia_3.0/cuda/heartwall"
logdir      = "logs/heartwall"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)


#
# Rodinia: Hotspots
#
program     = "../rodinia_3.0/cuda/hotspot"
logdir      = "logs/hotspot"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)


#
# Rodinia: Hybrid sort
#
#program     = "../rodinia_3.0/cuda/hybridsort"
#logdir      = "logs/hybridsort"
#arguments   = " "
#steps       = ["default"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
#
# Rodinia: K-Means
#
#program     = "../rodinia_3.0/cuda/kmeans"
#logdir      = "logs/kmeans"
#arguments   = " "
#steps       = ["default"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
#
# Rodinia: Lava Molecular Dynamics
#
#program     = "../rodinia_3.0/cuda/lavaMD"
#logdir      = "logs/lavaMD"
#arguments   = " "
#steps       = ["default"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
#
# Rodinia: LU Decomposition
#
program     = "../rodinia_3.0/cuda/lud"
logdir      = "logs/lud"
arguments   = " "
steps       = ["default"]

run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)


#
# Rodinia: Myocyte
#
#program     = "../rodinia_3.0/cuda/myocyte"
#logdir      = "logs/myocyte"
#arguments   = " "
#steps       = ["default"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
#
# Rodinia: KNN
#
#program     = "../rodinia_3.0/cuda/nn"
#logdir      = "logs/nn"
#arguments   = " "
#steps       = ["default"]
#
#run(program, steps, arguments, logdir, run_time, runs, benchmark, cuda_path, args)
#
#
# TODO: Write code for the other experiments.
#
