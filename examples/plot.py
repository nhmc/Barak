from barak.plot import distplot
import numpy as np

def test_distplot(n=100):
    a = [np.random.randn(n) for i in range(10)]
    x = range(len(a))
    ax = distplot(x,a, showoutliers=1)
    ax = distplot(x,a, color='r')
    ax = distplot(x,a, label='points', color='b', showmean=1)
    ax.legend()
