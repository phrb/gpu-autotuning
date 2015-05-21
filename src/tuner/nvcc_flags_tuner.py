import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

import argparse
import logging

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
NVCC_PARAMS = { "nvcc:--gpu-architecture="        : [ "sm_20", "sm_21",
                                                      "sm_30", "sm_32", "sm_35" ],
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

NVCC_CMD = "nvcc "

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

        compile_result = self.call_program(self.parse_config(cfg))
        assert compile_result['returncode'] == 0

        run_result = self.call_program("./tmp.bin " + " ".join(FARGS))
        assert run_result['returncode'] == 0

        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        cfg = configuration.data

        print "Optimal configuration written to 'final_config.json'."
        self.manipulator().save_to_file(cfg, LOG_DIR + 'final_config.json')

        cmd = self.parse_config(cfg)
        print "Optimal config written to " + LOG_DIR + LOG_FILE + ": ", cmd
        with open(LOG_DIR + LOG_FILE, 'a+') as myfile:
            myfile.write(cmd + "\n")

if __name__ == '__main__':
    opentuner.init_logging()
    args     = argparser.parse_args()

    LOG_DIR  = args.logdir
    LOG_FILE = args.logcmd

    filename = args.filename
    if (args.fargs):
        FARGS = args.fargs

    NVCC_CMD += "-w -I /usr/local/cuda/include -L /usr/local/cuda/lib64 " + filename + " -o ./tmp.bin "
    NvccFlagsTuner.main(argparser.parse_args())
