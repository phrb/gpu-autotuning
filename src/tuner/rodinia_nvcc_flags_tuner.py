import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

import subprocess
import argparse
import logging
import time
import os

import pandas

log = logging.getLogger('nvccflags')

argparser = argparse.ArgumentParser(parents=opentuner.argparsers())
argparser.add_argument( "-f", "--file",
                        dest     = "filename",
                        type     = str,
                        required = True,
                        help     = "A file to tune.")
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
argparser.add_argument( "-cp", "--cuda-path",
                        dest     = "cuda_path",
                        type     = str,
                        required = True,
                        help     = "Path to CUDA libraries.")

class NvccFlagsTuner(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for flag in NVCC_FLAGS + PTXAS_FLAGS + NVLINK_FLAGS:
            manipulator.add_parameter(
                    EnumParameter(flag,
                                  ["on", "off"]))

        for d in [NVCC_PARAMS, PTXAS_PARAMS]:
            for flag, values in d.iteritems():
                manipulator.add_parameter(EnumParameter(flag, values))

        for param, pmin, pmax in NVCC_NUM_PARAMS + PTXAS_NUM_PARAMS:
            manipulator.add_parameter(IntegerParameter(param, pmin, pmax))
        return manipulator

    def parse_flags(self, flag_list, target):
        cmd = ""
        for full_flag, value in flag_list:
            flag = full_flag.split(":")[1]
            if (value == "on"):
                cmd += " " + target + " " + flag + " "
            elif (value != "off"):
                cmd += " " + target + " " + flag + str(value) + " "

        return cmd

    def parse_config(self, cfg):
        nvcc_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("nvcc") ]
        ptxas_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("ptxas") ]
        nvlink_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("nvlink") ]

        cmd = NVCC_CMD
        cmd += self.parse_flags(nvcc_flags, "")
        cmd += self.parse_flags(ptxas_flags, PTXAS_NAME)
        cmd += self.parse_flags(nvlink_flags, NVLINK_NAME)

        return cmd

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        os.environ["NVCC_FLAGS"] = ""
        old_path = os.getcwd()
        os.chdir(FILENAME)
        subprocess.call("make clean", shell = True)
        subprocess.call("rm -f *.o *~ *.linkinfo", shell = True)

        os.environ["NVCC_FLAGS"] = self.parse_config(cfg)
        compile_result = subprocess.call("make", shell = True)
        os.chdir(old_path)
        if compile_result != 0:
            return Result(time=FAIL_PENALTY)

        # Give a better value to the Tuner (average)
        results = []
        evaluations = 10

        global CONFIGS_TESTED
        global results_df

        CONFIGS_TESTED += 1

        for i in range(evaluations):
            old_path = os.getcwd()
            os.chdir(FILENAME)
            run_result = self.call_program("./run")
            os.chdir(old_path)

            if run_result['returncode'] != 0:
                with open(LOG_DIR + "failed_configurations.txt", "a+") as file:
                    file.write("failed_example_cmd: " + self.parse_config(cfg) + "\n")
                global CONFIGS_FAILED
                CONFIGS_FAILED += 1

                results_df = results_df.append(pandas.DataFrame({ "response" : [FAIL_PENALTY],
                                                                  "run_id" : [CONFIGS_TESTED] }),
                                               ignore_index = True)

                return Result(time=FAIL_PENALTY)
            else:
                results_df = results_df.append(pandas.DataFrame({ "response" : [run_result['time']],
                                                                  "run_id" : [CONFIGS_TESTED] }),
                                               ignore_index = True)

                results.append(run_result['time'])

        results_df.to_csv(args.logdir + "measurements.csv", index = False)
        final_result = (sum(results) / len(results))
        return Result(time=final_result)

    def save_final_config(self, configuration):
        cfg = configuration.data

        print "Optimal configuration written to 'final_config.json'."
        self.manipulator().save_to_file(cfg, LOG_DIR + 'final_config.json')

        cmd = self.parse_config(cfg)
        print "Optimal config written to " + LOG_DIR + LOG_FILE + ": ", cmd
        with open(LOG_DIR + LOG_FILE, "a+") as myfile:
            myfile.write(cmd + "\n")

        with open(LOG_DIR + "failed_stats.txt", "a+") as file:
            file.write("tested_configurations             : " + str(CONFIGS_TESTED) + "\n")
            file.write("failed_configurations             : " + str(CONFIGS_FAILED) + "\n")
            file.write("ratio (% of failed configurations): " + str(CONFIGS_FAILED / float(CONFIGS_TESTED)) + "\n")

        global results_df
        results_df.to_csv(args.logdir + "measurements.csv", index = False)

if __name__ == '__main__':
    FARGS = ""
    # Specify gcc options:
    GCC_NAME = "-Xcompiler"
    GCC_FLAGS = []
    GCC_PARAMS = []
    GCC_NUM_PARAMS = []
    # name
    NVCC_FLAGS = [ "nvcc:--no-align-double",
                   "nvcc:--use_fast_math" ]
    # { name : [ args ] }
    # NVCC_PARAMS = { "nvcc:--gpu-architecture="        : [ "sm_60", "sm_61",
    #                                                       "sm_62", "sm_70",
    #                                                       "sm_72", "sm_75",
    #                                                       "sm_80", "sm_86"],
    NVCC_PARAMS = { "nvcc:--gpu-architecture="        : [ "sm_50"],
                    "nvcc:--relocatable-device-code=" : [ "true", "false" ],
                    "nvcc:--ftz="                     : [ "true", "false" ],
                    "nvcc:--prec-div="                : [ "true", "false" ],
                    "nvcc:--prec-sqrt="               : [ "true", "false" ] }
    # (name, min, max)
    NVCC_NUM_PARAMS = [ ]
    # Specify ptxas options:
    PTXAS_NAME = "-Xptxas "
    # name
    PTXAS_FLAGS  = [ ]
    # { name : [ args ] }
    PTXAS_PARAMS = { "ptxas:--def-load-cache="                : [ "ca", "cg", "cv", "cs" ],
                     "ptxas:--opt-level="                     : [ "0", "1", "2", "3" ],
                     "ptxas:--fmad="                          : [ "true", "false" ],
                     "ptxas:--allow-expensive-optimizations=" : [ "true", "false" ] }
    # ( name, min, max )
    PTXAS_NUM_PARAMS = [ ( "ptxas:--maxrregcount=", 16, 63 ) ]
    # Specify NVLINK options:
    NVLINK_NAME = "-Xnvlink "
    # name
    NVLINK_FLAGS = [ "nvlink:--preserve-relocs" ]

    FAIL_PENALTY   = 9999
    CONFIGS_FAILED = 0
    CONFIGS_TESTED = 0

    results_df = pandas.DataFrame({"response" : [],
                                   "run_id"   : [] })

    opentuner.init_logging()
    args     = argparser.parse_args()

    LOG_DIR  = args.logdir
    LOG_FILE = args.logcmd
    NVCC_CMD = ""
    CUDA_PATH = args.cuda_path

    FILENAME = args.filename
    if (args.fargs):
        FARGS = args.fargs

    NvccFlagsTuner.main(argparser.parse_args())
