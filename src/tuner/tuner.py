import opentuner 
from opentuner import ConfigurationManipulator 
from opentuner import EnumParameter 
from opentuner import IntegerParameter 
from opentuner import MeasurementInterface 
from opentuner import Result

GCC_FLAGS = [ "" ] 

# (name, min, max)
GCC_PARAMS = [
    ]
	
class GccFlagsTuner(MeasurementInterface):
    def manipulator(self):
        """
        Define the search space by creating a ConfigurationManipulator
        """



if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    GccFlagsTuner.main(argparser.parse_args())
