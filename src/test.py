from tqdm import tqdm
from time import sleep

points = 100

pbar = tqdm(
        total=points, ncols=110, desc="Loop Prog", leave=True, position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                   '[Elapsed: {elapsed}, To Go: {remaining}]{postfix}')
i=0


b=0
c=1
while i < points:
    try:
        if i != 43:
            c=1
        elif b == 0:
            b = 1
            c=0
        print(10/c)
        c = 1
        pbar.update()
        i += 1 
    except:
        continue
    
    sleep(0.1)

sleep(10)
pbar.close
