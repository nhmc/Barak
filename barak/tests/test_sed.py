from ..sed import *
from ..constants import Jy, c
import pylab as pl
import numpy as np

def test_seds():
    fors_u, fors_g, fors_r = get_bands('FORS','u,g,r',ccd='blue')
    sdss_u, sdss_g, sdss_r = get_bands('SDSS','u,g,r')
    
    pickles = get_SEDs('pickles')

    p_umg = [p.calc_colour(sdss_u, sdss_g, 'AB') for p in pickles]
    p_gmr = [p.calc_colour(sdss_g, sdss_r, 'AB') for p in pickles]

    tLBG = get_SEDs('LBG', 'lbg_em.dat')
    tLBGa = get_SEDs('LBG', 'lbg_abs.dat')
    tLBG_umg, tLBG_gmr = [], []
    tLBGa_umg, tLBGa_gmr = [], []
    for z in np.arange(2.2, 3.7, 0.2):
        tLBG.redshift_to(z)
        tLBGa.redshift_to(z)
        tLBG_umg.append(tLBG.calc_colour(sdss_u,sdss_g, 'AB'))
        tLBG_gmr.append(tLBG.calc_colour(sdss_g,sdss_r, 'AB'))
        tLBGa_umg.append(tLBGa.calc_colour(sdss_u,sdss_g, 'AB'))
        tLBGa_gmr.append(tLBGa.calc_colour(sdss_g,sdss_r, 'AB'))

def test_fnu_flambda():
    wa = np.logspace(1, 10, 1e5)
    fnu = 3631 * Jy
    assert np.allclose(fnu_to_flambda(wa, fnu), c*1e-8 * fnu / (wa * 1e-8)**2)
    assert np.allclose(fnu, flambda_to_fnu(wa, fnu_to_flambda(wa, fnu)))

def test_make_constant_dv_wa_scale():

    wa = make_constant_dv_wa_scale(1200, 1201, 8.8)
    assert np.allclose(wa, np.array(
        [ 1200.        ,  1200.03522437,  1200.07044977,  1200.10567621,
        1200.14090368,  1200.17613218,  1200.21136172,  1200.24659229,
        1200.2818239 ,  1200.31705654,  1200.35229022,  1200.38752493,
        1200.42276067,  1200.45799745,  1200.49323526,  1200.52847411,
        1200.56371399,  1200.5989549 ,  1200.63419685,  1200.66943984,
        1200.70468386,  1200.73992891,  1200.775175  ,  1200.81042212,
        1200.84567028,  1200.88091947,  1200.9161697 ,  1200.95142096]))
