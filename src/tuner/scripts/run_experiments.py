#! /usr/bin/python2

import subprocess
import argparse
import time
import os

argparser = argparse.ArgumentParser()
argparser.add_argument( "-f", "--file",
                        dest     = "filename",
                        type     = str,
                        required = True,
                        help     = "A file to run experiments in.")
argparser.add_argument( "-fargs", "--file-args",
                        dest     = "fargs",
                        type     = str,
                        nargs    = '*',
                        help     = "Program arguments.")
argparser.add_argument( "-ld", "--log-dir",
                        dest     = "logdir",
                        type     = str,
                        required = True,
                        help     = "Directory to save this tuning run.")
argparser.add_argument( "-lc", "--log-cmd",
                        dest     = "logcmd",
                        type     = str,
                        required = True,
                        help     = "File to save best configuration to.")
argparser.add_argument( "-time", "--run-time",
                        dest     = "time",
                        type     = str,
                        required = True,
                        help     = "Time to tune the program.")
argparser.add_argument( "-th", "--threads",
                        dest     = "parallelism",
                        type     = str,
                        required = True,
                        help     = "Number of threads to run the tuning program.")
argparser.add_argument( "-r", "--tuning-runs",
                        dest     = "runs",
                        type     = int,
                        required = True,
                        help     = "Number of tuning runs to perform.")
argparser.add_argument( "-br", "--benchmark-runs",
                        dest     = "benchmark",
                        type     = int,
                        required = True,
                        help     = "Number of times to run the final configuration.")
argparser.add_argument( "-tech", "--choose-technique",
                        dest     = "technique",
                        type     = str,
                        default  = "",
                        required = False,
                        help     = "Force OpenTuner to use an ensemble or specific technique.")
argparser.add_argument( "-s", "--seed",
                        dest     = "seed",
                        type     = str,
                        default  = "",
                        required = False,
                        help     = "A seed configuration, to start the search in.")
argparser.add_argument( "-cp", "--cuda-path",
                        dest     = "cuda_path",
                        type     = str,
                        default  = "",
                        required = True,
                        help     = "The path for CUDA libraries.")

if __name__ == '__main__':
    args =  argparser.parse_args()

    for i in range(args.runs):
        cmd    = "python2 nvcc_flags_tuner.py --no-dups"
        run_id = "/run_" + str(i)
        log_path = args.logdir + run_id
        subprocess.call("mkdir " + "'" + log_path + "'", shell = True)

        cmd += " --stop-after="         + args.time
        cmd += " --file="               + args.filename
        cmd += " --file-args="          + "\"" + " ".join(args.fargs) + "\""
        cmd += " --log-dir="            + log_path + "/"
        cmd += " --log-cmd="            + args.logcmd
        cmd += " --parallelism="        + args.parallelism
        cmd += " --results-log-detail=" + log_path + "/logall.txt"
        cmd += " --results-log="        + log_path + "/logbest.txt"
        cmd += " --cuda-path="          + "\"" + args.cuda_path + "\""

        if args.technique != "":
            cmd += " --technique=" + args.technique
        if args.seed != "":
            cmd += " --seed-configuration=" + args.seed

        subprocess.call(cmd, shell = True)

        #
        # Compile and run best solution multiple times.
        #
        print "[INFO] Tuning Complete, Starting Benchmark:"
        # Compiling:
        print "[INFO] Compiling..."
        subprocess.call("cat " + log_path + "/final", shell = True)
        subprocess.call("sh "  + log_path + "/final", shell = True)
        print "[INFO] Done."
        print "[INFO] Running Benchmark:"
        for j in range(args.benchmark):
            # Running:
            cmd     = "./tmp.bin " + " ".join(args.fargs)
            logfile = log_path + "/benchmark.txt"

            start   = time.time()
            subprocess.call(cmd, shell = True)
            end     = time.time()

            with open(logfile, "a+") as file:
                file.write(str(end - start) + "\n")


        print "[INFO] Benchmark Done."

        subprocess.call("rm -r opentuner.log opentuner.db", shell = True)

