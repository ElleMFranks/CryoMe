from dataclasses import dataclass
import numpy as np

# region Exception types.
class ConfigError(Exception):
    pass

class ParameterError(ConfigError):

    def __init__(self, parameter: str) -> None:
        error_msg = f'{parameter} is incorrect'
        ConfigError.__init__(self, error_msg)

# endregion

print('The FitnessGram PACER Test is a multistage aerobic capacity test that'
      ' progressively gets more difficult as it continues. The test is used'
      ' to measure a students aerobic capacity as part of the FitnessGram'
      ' assessment. Students run back and forth as many times as they can, '
      'each lap signaled by a beep sound.')

raise ConfigError('abcd')

raise ParameterError('xis')


@dataclass()
class Bias:
    gv: float
    dv: float
    di: float

amy = [Bias(0, 0.6, 4),
       Bias(0, 0.6, 6),
       Bias(0, 0.6, 8),
       Bias(0, 0.6, 10),
       Bias(0, 0.8, 4),
       Bias(0, 0.8, 6),
       Bias(0, 0.8, 8),
       Bias(0, 0.8, 10),
       Bias(0, 1, 4),  
       Bias(0, 1, 6),  
       Bias(0, 1, 8),  
       Bias(0, 1, 10)]


xxi = []

[xxi.append(bias.dv) for bias in amy if bias.di == 8]

print(xxi)


dvs = []
for bias in amy:
    dvs.append(bias.dv)

dvsset = set(dvs)

dv_array_of_arrays = []
for dv in dvsset:
    dv_array = []
    for bias in amy:
        if bias.dv == dv:
            dv_array.append(bias)
    dv_array_of_arrays.append(dv_array)

for array in dv_array_of_arrays:
    print(array)
    print('')
            
    
for a in amy:
    print(a)

print('')

amysort = sorted(amy, key=lambda bias: bias.di)

for a in amysort:
    print(a)