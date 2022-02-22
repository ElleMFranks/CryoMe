from tqdm import tqdm
from time import sleep
#import progressbar as pb

#for _ in tqdm(range(100), ncols=100):
#    sleep(0.3)

#bar = pb.ProgressBar(max_value=100, custom_len=100).start()

b = [0,0,0,0,0]
c = [0,1,0,0,0]
a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2,1,2,3,4,1,2,3,4,1,4,5,6,3,2]

try:
    for i, val in enumerate(b):
        try:
            x = 1/b[i]
            
        except:
            print(i)
            if i == 4:
                raise Exception('')

except:
    raise Exception('')

input('asdv')

for j in range(10):
    for i, k in enumerate(tqdm(a, ncols=110,
                               desc="Overall Prog",
            bar_format= '{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}, To Go: {remaining}{postfix}]')):
        sleep(0.1)

print('')
print('')