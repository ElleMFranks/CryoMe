from dataclasses import dataclass
import numpy as np

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