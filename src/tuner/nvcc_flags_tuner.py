import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

import argparse

argparser = argparse.ArgumentParser(parents=opentuner.argparsers())
argparser.add_argument( "-f", "--file",
                        dest = "filename",
                        required = True,
                        help = "A file to tune.")

# Specify compiler options:
NVCC_NAME = "-Xcompiler"
# name
NVCC_FLAGS = [ "nvcc:--no-align-double",
               "nvcc:--relocatable-device-code",
               "nvcc:--use_fast_math",
               "nvcc:--ftz",
               "nvcc:--prec-div",
               "nvcc:--prec-sqrt",
               "nvcc:--fmad" ]
# { name : [ args ] }
NVCC_PARAMS = { "nvcc:--default-stream="   : [ "legacy", "null", "per-thread" ],
                "nvcc:--gpu-architecture=" : [ "sm_20", "sm_21", "sm_30", "sm_32",
                                          "sm_35", "sm_50", "sm_52" ] }
# (name, min, max)
NVCC_NUM_PARAMS = [ ( "nvcc:--maxrregcount=", 16, 63 ) ]
# Specify ptxas options:
PTXAS_NAME = "-Xptxas "
# name
PTXAS_FLAGS  = [ "ptxas:--allow-expensive-optimizations",
                 "ptxas:--def-store-cache",
                 "ptxas:--disable-optimizer-consts",
                 "ptxas:--force-load-cache",
                 "ptxas:--force-store-cache",
                 "ptxas:--fmad" ]
# { name : [ args ] }
PTXAS_PARAMS = { "ptxas:--def-load-cache=" : [ "ca", "cg", "cv", "cs" ],
                 "ptxas:--gpu-name="       : [ "compute_20", "compute_30",
                                         "compute_35", "compute_50",
                                         "compute_52", "sm_20",
                                         "sm_21", "sm_30", "sm_32",
                                         "sm_35", "sm_50", "sm_52" ],
                 "ptxas:--opt-level="      : [ "0", "1", "2", "3" ] }
# ( name, min, max )
PTXAS_NUM_PARAMS = [ ( "ptxas:--maxrregcount=", 16, 63 ) ]
# Specify NVLINK options:
NVLINK_NAME = "-Xnvlink "
# name
NVLINK_FLAGS = [ "nvlink:--preserve-relocs" ]

NVCC_CMD = "nvcc "
NVCC_END = "-o "

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

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        nvcc_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("nvcc") ]
        ptxas_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("ptxas") ]
        nvlink_flags = [ (key, value) for key,value in cfg.iteritems() if key.startswith("nvlink") ]

        cmd = NVCC_CMD + NVCC_NAME
        for full_flag, value in nvcc_flags:
            flag = full_flag.split(":")[1]
            if (value == "on"):
                cmd += " " + flag + " "
            elif (value != "off"):
                cmd += " " + flag + str(value) + " "

        cmd += PTXAS_NAME
        for full_flag, value in ptxas_flags:
            flag = full_flag.split(":")[1]
            if (value == "on"):
                cmd += " " + flag + " "
            elif (value != "off"):
                cmd += " " + flag + str(value) + " "

        cmd += NVLINK_NAME
        for full_flag, value in nvlink_flags:
            flag = full_flag.split(":")[1]
            if (value == "on"):
                cmd += " " + flag + " "
            elif (value != "off"):
                cmd += " " + flag + str(value) + " "

        cmd += NVCC_END
        print cmd

#        return Result(time=0)
        return None

if __name__ == '__main__':
    args = argparser.parse_args()

    NVCC_END += args.filename
    NvccFlagsTuner.main(argparser.parse_args())
