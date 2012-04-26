from ..pyvpfit import *

def test_readatom():
    global atomdat
    atomdat = readatom(molecules=False)
    atomdat = readatom()

def test_findtrans():
    name, tr = findtrans('CIV 1550', atomdat)
    assert np.allclose(tr['wa'], 1550.7812)

