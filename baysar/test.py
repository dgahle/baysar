from time import sleep, time
from numpy import round, concatenate

def foo(a, b=None):
    sleep(a)
    print(a, b)
    return [a, b]

num=3

from multiprocessing import cpu_count

if num >= cpu_count():
    raise ValueError('Asked for more CPUs than availible!')

from multiprocessing import cpu_count, Process, Pipe, Event, Pool

pools=Pool(num)

start_time=time()

result=pools.map(foo, (num for n in range(num)))

t=time()-start_time

print('Took {} s to go {} s.'.format(round(t, 2), num**2))
print(result)
# p.terminate()

# import concurrent.futures
#
# start_time=time()
# with concurrent.futures.ProcessPoolExecutor(max_workers=num) as executor:
#     out=executor.map(foo, (num for n in range(num)))
#     res=concatenate([a for a in out])
#
# t=time()-start_time
#
# print('Took {} s to go {} s.'.format(round(t, 2), num**2))
# print(res)
