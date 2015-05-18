import opentuner
from opentuner import ConfigurationManipulator
from opentuner import EnumParameter
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result

# Specify compiler options:
NVCC_NAME = "-Xcompiler"
# name
NVCC_FLAGS = [ "--no-align-double=",
               "--relocatable-device-code=",
               "--use_fast_math=",
               "--ftz=",
               "--prec-div=",
               "--prec-sqrt=",
               "--fmad " ]
# { name : [ args ] }
NVCC_PARAMS = { "--default-stream="   : [ "legacy", "null", "per-thread" ],
                "--gpu-architecture=" : [ "sm_20", "sm_21", "sm_30", "sm_32",
                                          "sm_35", "sm_50", "sm_52" ] }
# (name, min, max)
NVCC_NUM_PARAMS = [ ( "--maxrregcount=", 16, 63 ) ]
# Specify ptxas options:
PTXAS_NAME = "-Xptxas "
# name
PTXAS_FLAGS  = [ "--allow-expensive-optimizations=",
                 "--def-store-cache=",
                 "--disable-optimizer-consts=",
                 "--force-load-cache=",
                 "--force-store-cache=",
                 "--fmad=" ]
# { name : [ args ] }
PTXAS_PARAMS = { "--def-load-cache=" : [ "ca", "cg", "cv", "cs" ],
                 "--gpu-name="       : [ "compute_20", "compute_30",
                                         "compute_35", "compute_50",
                                         "compute_52", "sm_20",
                                         "sm_21", "sm_30", "sm_32",
                                         "sm_35", "sm_50", "sm_52" ],
                 "--opt-level="      : [ "0", "1", "2", "3" ] }
# ( name, min, max )
PTXAS_NUM_PARAMS = [ ( "_ptxas_--maxrregcount=", 16, 63 ) ]
# Specify NVLINK options:
NVLINK_NAME = "-Xnvlink "
# name
NVLINK_FLAGS = [ "--preserve-relocs=" ]

class NvccFlagsTuner(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        for flag in NVCC_FLAGS + PTXAS_FLAGS + NVLINK_FLAGS:
            manipulator.add_parameter(
                    EnumParameter(flag,
                                  ["on", "off", "default"]))

        for d in [NVCC_PARAMS, PTXAS_PARAMS]:
            for flag, values in d.iteritems():
                manipulator.add_parameter(EnumParameter(flag, values))

        for param, pmin, pmax in NVCC_NUM_PARAMS + PTXAS_NUM_PARAMS:
            manipulator.add_parameter(IntegerParameter(param, pmin, pmax))
        return manipulator

    def run(self, desired_result, input, limit):
        print desired_result.configuration.data
        return Result(time=0)

if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    NvccFlagsTuner.main(argparser.parse_args())
