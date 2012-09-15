from ..absorb import *
from ..utilities import get_data_path
import numpy as np

DATAPATH = get_data_path()

def test_readatom():
    global atomdat
    atomdat = readatom(molecules=False)
    atomdat = readatom()

def test_calctau():    
    wav0,osc,gam = 1215.6701,0.4164,6.265E8   # Ang, unitless, s^-1
    btemp,bturb = 20., 0.                     # km/s
    v = np.linspace(-100, 100, 500)           # km/s
    logN = 13.0                               # cm^-2
    tau = calctau(v, wav0, osc, gam, logN, btemp=btemp, bturb=bturb)
    tau13 = np.loadtxt(DATAPATH + 'tests/tau_n13.txt.gz')
    assert np.allclose(tau, tau13)
    
    v = np.linspace(-1000, 1000, 1000)        # km/s
    logN = 21.                                # cm^-2
    tau = calctau(v, wav0, osc, gam, logN, btemp=btemp, bturb=bturb)
    tau21 = np.loadtxt(DATAPATH + 'tests/tau_n21.txt.gz')
    assert np.allclose(tau, tau21)
    
def text_calc_iontau():
    wa = np.linspace(2500, 2700, 5000)
    tau = calc_iontau(wa, atom['CIV'], 1.7, 14, 50)
    assert abs(tau.max() - 0.8803955) < 1e-6
    assert abs(tau.min() - 0.0) < 1e-6

def test_findtrans():
    atomdat = readatom()
    name, tr = findtrans('CIV 1550', atomdat)
    assert np.allclose(tr['wa'], 1550.7812)

def test_tau_LL():
    assert np.allclose(
        tau_LL(18, np.linspace(800, 912, 10)),
        [ 4.25502044, 4.4566929, 4.66463913, 4.8789552, 5.09973723,
          5.3270813, 5.56108352, 5.80183997, 6.04944677, 0. ])
