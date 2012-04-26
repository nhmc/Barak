import numpy as np
from ..voigt import *
from ..utilities import get_data_path

a = 0.01
u = np.linspace(-25, 25, 5000)
vtest = np.loadtxt(get_data_path() + 'tests/idl_voigt.txt.gz')

def test_voigt():
    assert np.allclose(voigt(a, u), vtest, rtol=1e-05, atol=1e-08)

def test_voigt_wofz():
    assert np.allclose(voigt_wofz(a, u), vtest, rtol=1e-05, atol=1e-08)
