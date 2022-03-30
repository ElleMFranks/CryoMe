import numpy as np

a = [1, 2, 3, 4, 5, 6]
b = [4, 3, 5, 6, 3, 5]
c = [1, 6, 3, 5, 4, 6]
d = [3, 4, 1 ,6 ,5 ,3]
e = [3, 5, 2, 4, 5, 6]

x = list([a, b, c, d, e])

print(x)

x = np.array(x)

print(x[:, 3])