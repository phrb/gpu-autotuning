#! /usr/bin/python2

import os
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

def tune(program, arguments, logdir, time, runs, benchmark, cuda_path):
    os.system("mkdir "   + logdir)
    cmd = "./scripts/run_experiments.py"
    cmd += " -f="        + program
    cmd += " -fargs="    + "\"" + " ".join(arguments) + "\""
    cmd += " -ld="       + logdir
    cmd += " -lc=final"
    cmd += " -time="     + str(time)
    cmd += " -th=1"
    cmd += " -r="        + str(runs)
    cmd += " -br="       + str(benchmark)
    cmd += " -cp="       + "\"" + cuda_path + "\""
    os.system(cmd)

def baseline(program, arguments, logdir, runs, cuda_path):
    options = "-Xptxas --opt-level="
    values  = ["0", "1", "2", "3"]

    os.system("mkdir " + logdir + "_baseline")
    for value in values:
        # Compiling:
        print "nvcc -w " + cuda_path + program + " -o tmp.bin " + options + value
        os.system("nvcc -w " + cuda_path + program + " -o tmp.bin " + options + value)
        for i in range(runs):
            time     = "/usr/bin/time -p "
            binary   = "./tmp.bin " + " ".join(arguments) + " "
            greptime = "2>&1 | grep -oP '(?<=real )[0-9]*.[0-9]*' "
            logfile  = ">> " + logdir + "_baseline/opt_" + value + ".txt"

            os.system(time + binary + greptime + logfile)

def run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args):
    print "[INFO] Starting " + program + " Experiments."
    os.system("mkdir " + logdir)
    for i in steps:
        print "[INFO] Size: " + str(i)
        if args.baseline == False:
            logs = logdir + "/size_" + str(i) + "_time_" + str(time)
            tune(program, [str(i), arguments], logs, time, runs, benchmark, cuda_path)
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
time        = 30
runs        = 1
benchmark   = 1
#
# SubSeqMax Experiments:
#
#program     = "../bioinformatic/SubSeqMax.cu"
#logdir      = "logs/SubSeqMax"
#arguments   = "0"
#steps       = [2**25, 2**26, 2**27, 2**28, 2**29, 2**30]
#
#run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args)
#
# MatMulShared Experiments:
#
#program     = "../matMul/matMul_gpu_sharedmem.cu"
#logdir      = "logs/MatMulShared"
arguments   = "32 0"
steps       = [128, 256, 512, 1024]

#run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args)
#
# MatMulSharedUn Experiments:
#
program     = "../matMul/matMul_gpu_sharedmem_uncoalesced.cu"
logdir      = "logs/MatMulSharedUn"

run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args)
#
# MatMulUn Experiments:
#
program     = "../matMul/matMul_gpu_uncoalesced.cu"
logdir      = "logs/MatMulUn"

run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args)
#
# MatMulGPU Experiments:
#
program     = "../matMul/matMul_gpu.cu"
logdir      = "logs/MatMulGPU"

run(program, steps, arguments, logdir, time, runs, benchmark, cuda_path, args)
#
# TODO: Write code for the other experiments.
#
