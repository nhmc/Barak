""" Functions and Classes used to fit an estimate of an unabsorbed
continuum to a QSO spectrum.
"""
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.transforms as mtran

from utilities import between, Gaussian, stats, indexnear
from convolve import convolve_psf
from io import loadobj, saveobj
from interp import AkimaSpline
from spec import qso_template_uv

import os

def spline_continuum(wa, fl, er, edges, minfrac=0.01, nsig=3.0,
                     resid_std=1.3, debug=False):
    """ Given a section of spectrum, fit a continuum to it very
    loosely based on the method in Aguirre et al. 2002.

    Parameters
    ----------
    wa               : Wavelengths.
    fl               : Fluxes.
    er               : One sigma errors.
    edges            : Wavelengths giving the chunk edges.
    minfrac = 0.01   : At least this fraction of pixels in a single chunk
                       contributes to the fit.
    nsig = 3.0       : No. of sigma for rejection for clipping.
    resid_std = 1.3  : Maximum residual st. dev. in a given chunk.
    debug = False    : If True, make helpful plots.

    Returns
    -------
    Continuum array and spline points

    """

    # Overview:

    # (1) Calculate the median flux value for each wavelength chunk.

    # (2) fit a 1st order spline (i.e. series of straight line
    # segments) through the set of points given by the central
    # wavelength for each chunk and the median flux value for each
    # chunk.

    # (3) Remove any flux values that fall more than nsig*er below
    # the spline.

    # Repeat 1-3 until the continuum converges on a solution (if it
    # doesn't throw hands up in despair! Essential to choose a
    # suitable first guess with small enough chunks).

    if len(edges) < 2:
        raise ValueError('must be at least two bin edges!')

    wa,fl,er = (np.asarray(a, np.float64) for a in (wa,fl,er))

    if debug:
        ax = pl.gca()
        ax.cla()
        ax.plot(wa,fl)
        ax.plot(wa,er)
        ax.axhline(0, color='0.7')
        good = ~np.isnan(fl) & ~np.isnan(er)
        ymax = 2*sorted(fl[good])[int(len(fl[good])*0.95)]
        ax.set_ylim(-0.1*ymax, ymax)
        ax.set_xlim(min(edges), max(edges))
        ax.set_autoscale_on(0)
        pl.draw()

    npts = len(wa)
    mask = np.ones(npts, bool)
    oldco = np.zeros(npts, float)
    co = np.zeros(npts, float)

    # find indices of chunk edges and central wavelengths of chunks
    indices = wa.searchsorted(edges)
    indices = [(i0,i1) for i0,i1 in zip(indices[:-1],indices[1:])]
    if debug:  print ' indices', indices
    wavc = [0.5*(w1 + w2) for w1,w2 in zip(edges[:-1],edges[1:])]

    # information per chunks
    npts = len(indices)
    mfl = np.zeros(npts, float)     # median fluxes at chunk centres
    goodfit = np.zeros(npts, bool)  # is fit acceptable?
    res_std = np.zeros(npts, float) # residuals standard dev
    res_med = np.zeros(npts, float) # residuals median
    if debug:
        print 'chunk centres', wavc
        cont, = ax.plot(wa,co,'k')
        midpoints, = ax.plot(wavc, mfl,'rx',mew=1.5,ms=8)

    # loop that iterative fits continuum
    while True:
        for i,(j1,j2) in enumerate(indices):
            if goodfit[i]:  continue
            # calculate median flux
            #print i,j1,j2
            w,f,e,m = (item[j1:j2] for item in (wa,fl,er,mask))
            ercond = (e > 0) & (~np.isnan(f))
            cond = m & ercond
            chfl = f[cond]
            chflgood = f[ercond]
            if len(chflgood) == 0: continue
            #print len(chfl), len(chflgood)
            if float(len(chfl)) / len(chflgood) < minfrac:
                f_cutoff = np.percentile(chflgood, minfrac)
                cond = ercond & (f >= f_cutoff)
            if len(f[cond]) == 0:  continue
            mfl[i] = np.median(f[cond])

        # calculate the spline. add extra points on either end to give
        # a nice slope at the end points.
        extwavc = ([wavc[0] - (wavc[1] - wavc[0])] + list(wavc) +
                   [wavc[-1] + (wavc[-1] - wavc[-2])])
        extmfl = ([mfl[0] - (mfl[1] - mfl[0])] + list(mfl) +
                  [mfl[-1] + (mfl[-1] - mfl[-2])])
        co = np.interp(wa, extwavc, extmfl)
        if debug:
            cont.set_ydata(co)
            midpoints.set_xdata(wavc)
            midpoints.set_ydata(mfl)
            pl.draw()

        # calculate residuals for each chunk
        for i,(j1,j2) in enumerate(indices):
            if goodfit[i]:  continue
            ercond = er[j1:j2] > 0
            cond = ercond & mask[j1:j2]
            chfl = fl[j1:j2][cond]
            chflgood = fl[j1:j2][ercond]
            if len(chflgood) == 0:  continue
            if float(len(chfl)) / len(chflgood) < minfrac:
                f_cutoff = np.percentile(chflgood, minfrac)
                cond = ercond & (fl[j1:j2] > f_cutoff)
            #print len(co), len(fl), i1, j1, j2
            residuals = (fl[j1:j2][cond] - co[j1:j2][cond]
                         ) / er[j1:j2][cond]
            res_std[i] = residuals.std()
            if len(residuals) == 0:
                continue
            res_med[i] = np.median(residuals)
            # If residuals have std < 1.0 and mean ~1.0, we might have
            # a reasonable fit.
            if res_std[i] <= resid_std:
                goodfit[i] = True

        if debug:
            print 'median and st. dev. of residuals by region - aiming for 0,1'
            for i,(f0,f1) in  enumerate(zip(res_med, res_std)):
                print '%s %.2f %.2f' % (i,f0,f1)
            raw_input('Enter...')

        # (3) Remove flux values that fall more than N*sigma below the
        # spline fit.
        cond = (co - fl) > nsig * er
        if debug:
            print np.nanmax(np.abs(co - oldco)/co)
        # Finish when the biggest change between the new and old
        # medians is smaller than the number below.
        if np.nanmax(np.abs(co - oldco)/co) < 4e-3:
            break
        oldco = co.copy()
        mask[cond] = False

    # finally fit a cubic spline through the median values to
    # get a smooth continuum.
    final = AkimaSpline(wavc, mfl)

    return final(wa), zip(wavc,mfl)


def fitqsocont(wa, fl, er, redshift, oldco=None, knots=None,
               nbin=1, divmult=1, forest_divmult=1, atmos=True, debug=False):

    """ Find an estimate of a QSO continuum.

    divmult=3 works well for R~40000, S/N~10, z=3 QSO spectrum.

    nbin bins the data for plotting and continuum fitting (obsolete)
    """    
    # choose initial reference continuum points.  Increase divmult for
    # fewer initial continuum points (generally needed for poorer S/N
    # spectra).

    zp1 = 1 + redshift
    #reflines = np.array([1025.72, 1215.6701, 1240.14, 1398.0,
    #                     1549.06, 1908,      2800            ])

    # generate the edges of wavelength chunks to send to fitting routine

    # these edges and divisions are generated by trial and error

    # for S/N = 15ish and resolution = 2000ish
    div = np.rec.fromrecords([(500. , 800. , 25),
                              (800. , 1190., 25),
                              (1190., 1213.,  4),
                              (1213., 1230.,  6),
                              (1230., 1263.,  6),
                              (1263., 1290.,  5),
                              (1290., 1340.,  5),
                              (1340., 1370.,  2),
                              (1370., 1410.,  5),
                              (1410., 1515.,  5),
                              (1515., 1600., 15),
                              (1600., 1800.,  8),
                              (1800., 1900.,  5),
                              (1900., 1940.,  5),
                              (1940., 2240., 15),
                              (2240., 3000., 25),
                              (3000., 6000., 80),
                              ], names='left,right,num')

    div.num[2:] = np.ceil(div.num[2:] * divmult)
    div.num[:2] = np.ceil(div.num[:2] * forest_divmult)    
    div.left *= zp1
    div.right *= zp1
    if debug: print div.tolist()
    temp = [np.linspace(left, right, n+1)[:-1] for left,right,n in div]
    edges = np.concatenate(temp)
    if debug: stats(edges)

    i0,i1,i2 = edges.searchsorted([wa[0], 1210*zp1, wa[-1]])
    if debug: print i0,i1,i2

    contpoints = []
    if knots is not None:
        contpoints.extend(knots)
    else:
        co,cp = spline_continuum(wa, fl, er, edges[i0:i2], debug=debug)
        contpoints.extend(cp)
    fig = pl.figure(figsize=(11, 7))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
    wrapper = InteractiveCoFit(wa, fl, er, contpoints, co=oldco, nbin=nbin,
                               redshift=redshift, fig=fig, atmos=atmos)
    while True:
        if wrapper.finished: break
        pl.waitforbuttonpress()

    return wrapper.continuum, wrapper.contpoints

class InteractiveCoFit(object):
    help_message = """
'a'        : add a new continuum point
'd'        : delete the nearest point
'b'        : add a break in the continuum
'r'        : remove a break in the continuum
's'        : smooth the spectrum
'k'        : keep continuum
'q'        : quit without keeping continuum
"""
    def __init__(self, wa, fl, er, contpoints, co=None,
                 nbin=8, redshift=None, atmos=None, fig=None):
        """ Initialise figure, plots and variables.

        Parameters
        ----------
        wa :   Wavelengths
        fl :   Fluxes
        er :   One sigma errors
        nbin : int (8)
            Number of pixels to bin arrays in wavelength. Default 8.
        contpoints : list of x,y tuple pairs (None)
            The points through which a cubic spline is passed,
            defining the continuum.
        redshift : float (None)
            Redshift used to plot reference emission lines.
        atmos : list of wavelength pairs (None)
            Regions of atmospheric absorption to plot.

        Notes
        -----
        Updates the following attributes:
        
         self.spec :  Dictionary of wa, fl, er.
         self.contpoints :  Points used to define the continuum.
         self.nbin :  The input nbin value.
         self.markers :  Dictionary of matplotlib plotting artists.
         self.connections :  Callback connections.
         self.fig :  The plotting figure instance.
        """
        #setup
        #print co
        self.WMIN_LYA = 1040
        self.WMAX_LYA = 1190

        self.spec = dict(wa=wa, fl=fl, er=er, co=co)
        self.nbin = nbin
        self.breaks = [wa[0], wa[-1]] # wavelengths of breaks in the continuum
        self.contpoints = list(contpoints)
        if os.path.lexists('./_knots.sav'):
            c = raw_input('temporary knots file exists, use these knots? (y) ')
            if c.lower() != 'n':
                self.contpoints = loadobj('./_knots.sav')

        self.markers = dict()
        self.art_fl = None
        if fig is None:
            self.fig = pl.figure()
        else:
            self.fig = fig
        # disable any existing key press callbacks
        cids = list(fig.canvas.callbacks.callbacks['key_press_event'])
        for cid in cids:
            fig.canvas.callbacks.disconnect(cid)

        self.template = None
        if redshift is not None:
            self.template = qso_template_uv(wa, redshift)

        self.connections = []
        self.continuum = None
        self.finished = False
        self.redshift = redshift
        self.atmos = atmos
        self.smoothby = None
        self.plotinit()
        self.update()
        self.modifypoints()
        pl.draw()

    def plotinit(self):
        """ Set up the figure and do initial plots.

        Updates the following attributes:

          self.markers
        """
        wa,fl,er = [self.spec[k][0:-1:self.nbin] for k in 'wa fl er'.split()]
        if self.spec['co'] is not None:
            co = self.spec['co'][0:-1:self.nbin]
        # axis for spectrum & continuum
        a0 = self.fig.add_axes((0.05,0.1,0.9,0.6))
        a0.set_autoscale_on(0)
        # axis for residuals
        a1 = self.fig.add_axes((0.05,0.75,0.9,0.2),sharex=a0)
        a1.set_autoscale_on(0)
        a1.axhline(0,color='k',alpha=0.7, zorder=99)
        a1.axhline(1,color='k',alpha=0.7, zorder=99)
        a1.axhline(-1,color='k',alpha=0.7, zorder=99)
        a1.axhline(2,color='k',linestyle='dashed',zorder=99)
        a1.axhline(-2,color='k',linestyle='dashed',zorder=99)
        m0, = a1.plot([0],[0],'.r', ms=6, alpha=0.5)
        a1.set_ylim(-4, 4)
        a0.axhline(0, color='0.7')
        if self.spec['co'] is not None:
            a0.plot(wa,co, color='0.7', lw=1, ls='dashed')
        self.art_fl, = a0.plot(wa, fl, 'b', lw=0.5, linestyle='steps-mid')
        a0.plot(wa, er, lw=0.5, color='orange')
        m1, = a0.plot([0], [0], 'r', alpha=0.7)
        m2, = a0.plot([0], [0], 'o', mfc='None',mew=1, ms=8, mec='r', picker=5,
                      alpha=0.7)
        a0.set_xlim(min(wa), max(wa))
        good = (er > 0) & ~np.isnan(fl)
        ymin = -5 * np.median(er[good])
        ymax = 2 * sorted(fl[good])[int(good.sum()*0.95)]
        a0.set_ylim(ymin, ymax)
        a0.text(0.9,0.9, 'z=%.2f' % self.redshift, transform=a0.transAxes)

        # for histogram
        trans = mtran.blended_transform_factory(a1.transAxes, a1.transData)
        hist, = a1.plot([], [], color='k', transform=trans)
        x = np.linspace(-3,3)
        a1.plot(Gaussian(x,0,1,0.05), x, color='k', transform=trans, lw=0.5)

        if self.template is not None:
            trans = mtran.blended_transform_factory(a0.transData, a0.transAxes)                
            a0.plot(self.spec['wa'], self.template/self.template.max(), '-c', lw=2,
                    alpha=0.5, transform=trans)

        self.fig.canvas.draw()
        self.markers.update(contpoints=m2, cont=m1, resid=m0, hist_left=hist)

    def update(self):
        """ Calculates the new continuum, residuals and updates the plots.


        Updates the following attributes:

          self.markers
          self.continuum
        """
        wa,fl,er = (self.spec[key] for key in 'wa fl er'.split())
        co = np.empty(len(wa))
        co.fill(np.nan)
        for b0,b1 in zip(self.breaks[:-1], self.breaks[1:]):
            cpts = [(x,y) for x,y in self.contpoints if b0 <= x <= b1]
            if len(cpts) == 0:
                continue 
            spline = AkimaSpline(*zip(*cpts))
            i,j = wa.searchsorted([b0,b1])
            co[i:j] = spline(wa[i:j])
        
        resid = (fl - co) / er
        # histogram
        bins = np.arange(0, 5+0.1, 0.2)
        w0,w1 = self.fig.axes[1].get_xlim()
        x,_ = np.histogram(resid[between(wa, w0, w1)],
                           bins=bins)
        b = np.repeat(bins, 2)
        X = np.concatenate([[0], np.repeat(x,2), [0]])
        Xmax = X.max()    
        X = 0.05 * X / Xmax
        self.markers['hist_left'].set_data(X, b)

        self.markers['contpoints'].set_data(zip(*self.contpoints))
        nbin = self.nbin
        self.markers['cont'].set_data(wa[::nbin], co[::nbin])
        self.markers['resid'].set_data(wa[::nbin], resid[::nbin])
        if self.smoothby is not None:
            sfl = convolve_psf(fl, self.smoothby)
            self.art_fl.set_data(wa, sfl)
        else:
            self.art_fl.set_data(wa, fl)
        self.continuum = co
        saveobj('_knots.sav', self.contpoints, overwrite=True)
        self.fig.canvas.draw()

    def on_keypress(self, event):
        """ Add or remove a continuum point.

        Updates:
        
         self.contpoints
        """
        if event.key == 'q':
            for item in self.connections:
                self.fig.canvas.mpl_disconnect(item)
            self.contpoints = None
            self.continuum = None
            self.finished = True
            return
        if event.key == 'k':
            for item in self.connections:
                self.fig.canvas.mpl_disconnect(item)
            self.finished = True
            return
        if event.inaxes != self.fig.axes[0]:  return
        
        if event.key == 'a':
            # add a point to contpoints
            x,y = event.xdata,event.ydata
            if x not in zip(*self.contpoints)[0]:
                self.contpoints.append((x,y))
                self.update()
        elif event.key == 'd':
            # remove a point from contpoints
            contx,conty = zip(*self.contpoints)
            sep = np.hypot(event.xdata - contx, event.ydata - conty)
            self.contpoints.remove(self.contpoints[sep.argmin()])
            self.update()
        elif event.key == 'b':
            # Add a break to the continuum.
            self.breaks.append(event.xdata)
            self.breaks.sort()
            self.update()
        elif event.key == 'r':
            # remove a break
            i = indexnear(self.breaks, event.xdata)
            if i not in (0, len(self.breaks)-1):
                self.breaks.remove(self.breaks[i])
            self.update()
        elif event.key == 's':
            c = raw_input('New FWHM in pixels of Gaussian to convolve with? '
                          '(blank for no smoothing) ')
            if c == '':
                # restore spectrum
                self.smoothby = None
                self.update()
            else:
                try:
                    fwhm = float(c)
                except TypeError:
                    print 'FWHM must be a floating point number >= 1'
                if fwhm < 1:
                    self.smoothby = None
                else:
                    self.smoothby = fwhm
                self.update()
        elif event.key == '?':
            print self.help_message

    def on_button_release(self, event):
        self.update()

    def modifypoints(self):
        """ Add/remove continuum points."""
        print self.help_message
        id1 = self.fig.canvas.mpl_connect('key_press_event',self.on_keypress)
        id2 = self.fig.canvas.mpl_connect('button_release_event',self.on_button_release)
        self.connections.extend([id1, id2])


