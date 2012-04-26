from ..utilities import *
import numpy as np

DATAPATH = get_data_path()

fl = np.linspace(0,1)

def test_possion_noise():
    fl0,er0 = poisson_noise(fl, 0.1, seed=114)
    fl1,er1 = np.loadtxt(DATAPATH + 'tests/noisepoisson.txt.gz', unpack=1)
    assert np.allclose(fl0,fl1), np.allclose(er0,er1)

def test_addnoise():
    fl0,er0 = addnoise(fl, 0.2, seed=113)
    fl1,er1 = np.loadtxt(DATAPATH + 'tests/noisegauss.txt.gz', unpack=1)
    assert np.allclose(fl0,fl1), np.allclose(er0, er1)

    fl0,er0 = addnoise(fl, 0.2, minsig=0.05, seed=116)
    fl1,er1 = np.loadtxt(DATAPATH + 'tests/noiseboth.txt.gz', unpack=1)
    assert np.allclose(fl0,fl1), np.allclose(er0, er1)

def test_wmean():
    val = np.concatenate([np.ones(100), np.ones(100)*2])
    sig = range(1, 201)
    mean,err = wmean(val, sig)
    assert np.allclose((mean, err), (1.003026102, 0.78088153553))

def test_indexnear():
    wa = np.linspace(4000, 4500, 100)
    assert indexnear(wa, 4302.5) == 60
    assert indexnear(wa, 4600.0) == 99
    assert indexnear(wa, 3000.0) == 0

def test_find_edges_true_regions():
    a = np.array([3,0,1,4,6,7,8,6,3,2,0,3,4,5,6,4,2,0,2,5,0,3])
    ileft, iright = find_edges_true_regions(a > 2)
    assert zip(ileft, iright) == [(0, 0), (3, 8), (11, 15),
                                  (19, 19), (21, 21)]

