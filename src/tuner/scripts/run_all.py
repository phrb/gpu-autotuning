#! /usr/bin/python2

import os

def run(program, arguments, logdir, time):
    os.system("mkdir " + logdir)
    cmd = "./scripts/run_experiments.py"
    cmd += " -f=" + program
    cmd += " -fargs=" + "\"" + " ".join(arguments) + "\""
    cmd += " -ld=" + logdir
    cmd += " -lc=final"
    cmd += " -time=" + str(time)
    cmd += " -th=1"
    cmd += " -r=1"
    cmd += " -br=0"
    os.system(cmd)

# SubSeqMax Experiments:
program = "../bioinformatic/SubSeqMax.cu"
logdir  = "logs/SubSeqMax"
time    = 900

print "[INFO] Starting SubSeqMax Experiments."
for i in [2**27, 2**28, 2**29, 2**30, 2**31, 2**32]:
    logs = logdir + "_size_" + str(i) + "_time_" + str(time)
    run(program, [str(i), "0"], logs, time)
print "[INFO] SubSeqMax Experiments Done."
