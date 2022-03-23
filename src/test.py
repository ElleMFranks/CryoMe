import multiprocessing
from time import sleep
from time import perf_counter

start = perf_counter()

def do_something(time_for_sleep, msg):
    print('Enter')
    sleep(time_for_sleep)
    print(f'{msg}')
    print('Return')

p1 = multiprocessing.Process(target=do_something, args=[1, 'pelb'])
p2 = multiprocessing.Process(target=do_something, args=[2, 'blep'])

if __name__ == '__main__':

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    finish = perf_counter()
    print(f'Finished in {round(finish-start, 2)} s')
