from ..sed import *
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
     
