from ..virial import *
import numpy as np
from astropy import cosmology

cosmo = cosmology.WMAP7

def test_find_rvT():
    vir = find_rvT(1e12, 0, cosmo=cosmo)
    assert np.allclose([vir.r, vir.v, vir.T],
                       [261.31830192516918, 128.30867425264995,
                        588367.36865584785], rtol=1e-4)

    r,v,T = find_rvT(1e12, [0.5, 1, 1.5, 2], cosmo=cosmo)
    assert np.allclose(
        r, [ 199.03298324,  156.93226803,  128.36495081,  108.1516528 ],
        rtol=1e-4)
    assert np.allclose(
        v, [ 147.02067255,  165.57120355,  183.0702194 ,  199.44555006],
        rtol=1e-4)
    assert np.allclose(
        T, [  772490.86648696,   979729.43115699,  1197765.90662732,
              1421625.63132854], rtol=1e-4)
