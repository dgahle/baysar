from timeit import timeit
setup='''
from scipy.signal import fftconvolve
n={}
from numpy import ones
a=ones(n)
b=ones(n)
'''
lengths=[i for i in range(500, 1050)]

t=[timeit('fftconvolve(a, b)', setup=setup.format(i), number=100) for i in lengths]

from tulasa.general import plot

plot(t, x=lengths)
