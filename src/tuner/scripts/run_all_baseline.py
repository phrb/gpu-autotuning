#! /usr/bin/python2

import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument( "-cp", "--cuda-path",
                        dest     = "cuda_path",
                        type     = str,
                        required = True,
                        help     = "The path for CUDA libraries.")

def run(program, arguments, logdir, time, runs, benchmark, cuda_path):
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
    cmd += " -cp="       + cuda_path
    os.system(cmd)

def baseline(program, arguments, logdir, runs, cuda_path):
    options = "-Xptxas --opt-level="
    values  = ["0", "1", "2", "3"]

    os.system("mkdir " + logdir + "/baseline")
    for value in values:
        # Compiling:
        os.system("nvcc -w " + cuda_path + program + "-o ./tmp.bin" + options + value)
        for i in runs:
            time     = "/usr/bin/time -p "
            binary   = "./tmp.bin " + " ".join(arguments) + " "
            greptime = "2>&1 | grep -oP '(?<=real )[0-9]*.[0-9]*' "
            logfile  = ">> " + logdir + "/baseline/opt_" + value + ".txt"

            os.system(time + binary + greptime + logfile)

#
# SubSeqMax Experiments:
#
# Parameters:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
program     = "../bioinformatic/SubSeqMax.cu"
logdir      = "logs/SubSeqMax"
time        = 7200
runs        = 2
benchmark   = 20
#
# Starting a clean experiment.
#
os.system("rm -r " + logdir)
os.system("mkdir " + logdir)

print "[INFO] Starting SubSeqMax Experiments."
for i in [2**25, 2**26, 2**27, 2**28, 2**29, 2**30]:
    logs = logdir + "/size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "0"], logs, time, runs, benchmark, cuda_path)
    print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
    baseline(program, [str(i), "0"], logdir, benchmark, cuda_path)
    print "[INFO] Baseline Calculation Done."
print "[INFO] SubSeqMax Experiments Done."
#
# MatMulShared Experiments:
#
# Parameters:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
program     = "../matMul/matMul_gpu_sharedmem.cu"
logdir      = "logs/MatMulShared"
time        = 7200
runs        = 2
benchmark   = 20
#
# Starting a clean experiment.
#
os.system("rm -r " + logdir)
os.system("mkdir " + logdir)

print "[INFO] Starting MatMulShared Experiments."
for i in [16, 32, 64, 128, 256, 512, 1024]:
    logs = logdir + "/size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "32", "0"], logs, time, runs, benchmark, cuda_path)
    print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
    baseline(program, [str(i), "0"], logdir, benchmark, cuda_path)
    print "[INFO] Baseline Calculation Done."
print "[INFO] MatMulShared Experiments Done."
#
# MatMulSharedUn Experiments:
#
# Parameters:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
program     = "../matMul/matMul_gpu_sharedmem_uncoalesced.cu"
logdir      = "logs/MatMulSharedUn"
time        = 7200
runs        = 2
benchmark   = 20
#
# Starting a clean experiment.
#
os.system("rm -r " + logdir)
os.system("mkdir " + logdir)

print "[INFO] Starting MatMulSharedUn Experiments."
for i in [16, 32, 64, 128, 256, 512, 1024]:
    logs = logdir + "/size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "32", "0"], logs, time, runs, benchmark, cuda_path)
    print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
    baseline(program, [str(i), "0"], logdir, benchmark, cuda_path)
    print "[INFO] Baseline Calculation Done."
print "[INFO] MatMulSharedUn Experiments Done."
#
# MatMulUn Experiments:
#
# Parameters:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
program     = "../matMul/matMul_gpu_uncoalesced.cu"
logdir      = "logs/MatMulUn"
time        = 7200
runs        = 2
benchmark   = 20
#
# Starting a clean experiment.
#
os.system("rm -r " + logdir)
os.system("mkdir " + logdir)

print "[INFO] Starting MatMulUn Experiments."
for i in [16, 32, 64, 128, 256, 512, 1024]:
    logs = logdir + "/size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "32", "0"], logs, time, runs, benchmark, cuda_path)
    print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
    baseline(program, [str(i), "0"], logdir, benchmark, cuda_path)
    print "[INFO] Baseline Calculation Done."
print "[INFO] MatMulUn Experiments Done."
#
# MatMulGPU Experiments:
#
# Parameters:
#
args        = argparser.parse_args()
cuda_path   = args.cuda_path
program     = "../matMul/matMul_gpu.cu"
logdir      = "logs/MatMulGPU"
time        = 7200
runs        = 2
benchmark   = 20
#
# Starting a clean experiment.
#
os.system("rm -r " + logdir)
os.system("mkdir " + logdir)

print "[INFO] Starting MatMulGPU Experiments."
for i in [16, 32, 64, 128, 256, 512, 1024]:
    logs = logdir + "/size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "32", "0"], logs, time, runs, benchmark, cuda_path)
    print "[INFO] Calculating Baselines for -O0, -O1, -O2, -O3."
    baseline(program, [str(i), "0"], logdir, benchmark, cuda_path)
    print "[INFO] Baseline Calculation Done."
print "[INFO] MatMulGPU Experiments Done."
#
# TODO: Write code for the other experiments.
0#
