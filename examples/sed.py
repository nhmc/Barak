from barak.sed import get_bands, get_SEDs
import pylab as pl
import numpy as np

fors_u, fors_g, fors_r = get_bands('FORS','u,g,r',ccd='blue')
sdss_u, sdss_g, sdss_r = get_bands('SDSS','u,g,r')
pl.figure()
for b in fors_u, fors_g, fors_r, sdss_u, sdss_g, sdss_r:
    b.plot()

pickles = get_SEDs('pickles')
fig = pl.figure()
fig.subplots_adjust(left=0.18)
for p in pickles:
    p.plot(log=1)
pl.title('Pickles stellar library')

p_umg = [p.calc_colour(sdss_u, sdss_g, 'AB') for p in pickles]
p_gmr = [p.calc_colour(sdss_g, sdss_r, 'AB') for p in pickles]

tLBG = get_SEDs('LBG', 'lbg_em.dat')
tLBGa = get_SEDs('LBG', 'lbg_abs.dat')
tLBG_umg, tLBG_gmr = [], []
tLBGa_umg, tLBGa_gmr = [], []
zlabels = []
for z in np.arange(2.2, 3.7, 0.2):
    zlabels.append(str(z))
    tLBG.redshift_to(z)
    tLBGa.redshift_to(z)
    tLBG_umg.append(tLBG.calc_colour(sdss_u,sdss_g, 'AB'))
    tLBG_gmr.append(tLBG.calc_colour(sdss_g,sdss_r, 'AB'))
    tLBGa_umg.append(tLBGa.calc_colour(sdss_u,sdss_g, 'AB'))
    tLBGa_gmr.append(tLBGa.calc_colour(sdss_g,sdss_r, 'AB'))
 
tLBG_umg,tLBG_gmr,tLBGa_umg, tLBGa_gmr = map(
    np.array, [tLBG_umg,tLBG_gmr,tLBGa_umg, tLBGa_gmr])
 
tLBG_umg = tLBG_umg.clip(-5, 5)
tLBG_gmr = tLBG_gmr.clip(-5, 5)
tLBGa_umg = tLBGa_umg.clip(-5, 5) 
tLBGa_gmr = tLBGa_gmr.clip(-5, 5)

pl.figure()
ax = pl.gca()
ax.plot(p_gmr, p_umg, '.', label='Pickles')
ax.plot(tLBG_gmr, tLBG_umg, 'x-r', label='LBG em')
ax.plot(tLBGa_gmr, tLBGa_umg, 'x-b', label='LBG abs')
for i in xrange(len(zlabels)):
    ax.text(tLBG_gmr[i], tLBG_umg[i], zlabels[i])

pl.xlabel('SDSS g-r')
pl.ylabel('SDSS u-g')
pl.legend()
pl.show()
