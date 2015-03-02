""" Calculations involving the equivalent width.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from ..spec import find_bin_edges
from ..utilities import adict
from math import sqrt

__all__ = ['N_from_Wr_linear', 'log10N_from_Wr', 'calc_Wr']


def N_from_Wr_linear(osc, wrest):
    """ Find the multiplier to convert from equivalent width to a
    column density.

    Assumes the transition is on the linear part of the curve of
    growth. See Draine,"Physics of the
    Interstellar and Intergalactic medium", ISBN 978-0-691-12214-4,
    chapter 9.

    Parameters
    ----------
    osc: float
       Transition oscillator strength.
    wrest: float
       Transition rest wavelength in Angstroms.    

    Returns
    -------
    mult: float
       Multiply this vale by the rest-frame equivalent width in
       Angstroms to give the column density in absorbers per square
       cm.
    """
    return 1.13e20 / (osc * wrest**2)


def log10N_from_Wr(Wr, wa0, osc):
    """ Find log10(Column density) from a rest frame equivalent width
    assuming optically thin.

    Parameters
    ----------
    Wr : float
       Rest frame equivalent width in Angstroms.
    wa0 : float
       Transition wavelength in Angstroms.
    osc: float
       Transition oscillator strength.

    Returns
    -------
    log10N : float
      log10(column density in cm^-2), or zero if the equivalent width
      is 0 or negative.
    """
    if not Wr > 0:
        return 0

    Nmult = N_from_Wr_linear(wa0, osc)
    return np.log10(Nmult * Wr)


def calc_Wr(i0, i1, wa, tr, ew=None, ewer=None, fl=None, er=None, co=None,
            cohi=None, colo=None, colo_sig=0.5, cohi_sig=0.5):
    """ Find the rest equivalent width of a feature, and column
    density assuming optically thin.

    You must give either fl, er and co, or ew and ewer.

    Parameters
    ----------
    i0, i1 : int
      Start and end indices of feature (inclusive).
    wa : array of floats, shape (N,)
      Observed wavelengths.
    tr : atom.dat entry
      Transition entry from an atom.dat array read by `readatom()`.
    ew : array of floats, shape (N,), optional
      Equivalent width per pixel.
    ewer : array of floats, shape (N,), optional
      Equivalent width 1 sigma error per pixel.
      with attributes wav (rest wavelength) and osc (oscillator strength).
    fl : array of floats, shape (N,), optional
      Observed flux.
    er : array of floats, shape (N,), optional
      Observed flux 1 sigma error.
    co : array of floats, shape (N,), optional
      Observed continuum.
    cohi : float (None)
      When calculating logN upper error, increase the continuum by
      this fractional amount. Only used if fl, er and co are also
      given.
    colo : float (None)
      When calculating logN lower error decrease the continuum by
      this fractional amount.  Only used if fl, er and co are also
      given.
    colo_sig : float (0.5)
      If not None, decrease the continuum by this many sigma to find
      lower logN. Only used if fl, er and co are also given. Takes
      precendence over colo.
    cohi_sig : float (0.5)
      If not None, increase the continuum by this many sigma to find
      upper logN. Only used if fl, er and co are also given. Takes
      precedence over cohi.

    Returns
    -------
    A dictionary with keys:

    ========= =========================================================
    logN      1 sigma low val, value, 1 sigma upper val
    Ndetlim   log N 1 sigma upper limit
    Wr        Rest equivalent width in same units as wa
    Wre       1 sigma error on rest equivalent width
    zp1       1 + redshift
    ngoodpix  number of good pixels contributing to the measurements
    Nmult     multiplier to get from equivalent width to column density
    saturated Are more than 50% of the pixels between limits saturated?
    ========= =========================================================
    """
    wa1 = wa[i0:i1+1]

    npts = i1 - i0 + 1
    # if at least this many points are saturated, then mark the
    # transition as saturated
    n_saturated_thresh = int(0.5 * npts)
    saturated = None

    if ew is None:
        assert fl is not None and er is not None and co is not None
        wedge = find_bin_edges(wa1)
        dw = wedge[1:] - wedge[:-1]
        ew1 = dw * (1 - fl[i0:i1+1] / co[i0:i1+1])
        ewer1 = dw * er[i0:i1+1] / co[i0:i1+1]
        ewer1[np.isnan(ewer1)] = 0
        if cohi_sig is not None:
            c = co[i0:+i1+1] + cohi_sig * er[i0:i1+1]
        else:
            c = (1 + cohi) * co[i0:i1+1]
        ew1hi = dw * (1 - fl[i0:i1+1] / c)
        ewer1hi = dw * er[i0:i1+1] / c
        if colo_sig is not None:
            c = co[i0:+i1+1] - colo_sig * er[i0:i1+1]
        else:
            c = (1 - colo) * co[i0:i1+1]
        ew1lo = dw * (1 - fl[i0:i1+1] / c)
        ewer1lo = dw * er[i0:i1+1] / c
        if (fl[i0:i1+1] < er[i0:i1+1]).sum() >= n_saturated_thresh:
            saturated = True
        else:
            saturated = False
    else:
        
        assert None not in (ew, ewer)
        ew1 = np.array(ew[i0:i1+1])
        ewer1 = np.array(ewer[i0:i1+1])
        ewer1[np.isnan(ewer1)] = 0

    # interpolate over bad values
    good = ~np.isnan(ew1) & (ewer1 > 0)
    if not good.any():
        return None
    if not good.all():
        ew1[~good] = np.interp(wa1[~good], wa1[good], ew1[good])
        ewer1[~good] = np.interp(wa1[~good], wa1[good], ewer1[good])
        if fl is not None:
            ew1hi[~good] = np.interp(wa1[~good], wa1[good], ew1hi[good])
            ewer1hi[~good] = np.interp(wa1[~good], wa1[good], ewer1hi[good])
            ew1lo[~good] = np.interp(wa1[~good], wa1[good], ew1lo[good])
            ewer1lo[~good] = np.interp(wa1[~good], wa1[good], ewer1lo[good])

    W = ew1.sum()
    We = sqrt((ewer1**2).sum())
    if fl is not None:
        Whi = ew1hi.sum()
        Wehi = sqrt((ewer1hi**2).sum())
        Wlo = ew1lo.sum()
        Welo = sqrt((ewer1lo**2).sum())

    zp1 = 0.5*(wa1[0] + wa1[-1]) / tr['wa']
    Wr = W / zp1
    Wre = We / zp1
    if fl is not None:
        Wrhi = Whi / zp1
        Wrehi = Wehi / zp1
        Wrlo = Wlo / zp1
        Wrelo = Welo / zp1

    # Assume we are on the linear part of curve of growth (will be an
    # underestimate if not). See Draine, "Physics of the
    # Interstellar and Intergalactic medium", ISBN 978-0-691-12214-4,
    # chapter 9.
    Nmult = N_from_Wr_linear(tr['osc'], tr['wa'])

    # 1 sigma detection limit
    detlim = np.log10(Nmult * Wre)

    logN = (log10(Nmult * Wr) if Wr > 0 else 0)
    if fl is not None:
        whi = Wrhi + Wrehi
        wlo = Wrlo - Wrelo
    else:
        whi = Wr + Wre
        wlo = Wr - Wre
    try:
        logNhi = (log10(Nmult * whi) if whi > 0 else 0)
        logNlo = (log10(Nmult * wlo) if wlo > 0 else 0)
    except ValueError:
        import pdb; pdb.set_trace()

    return adict(logN=(logNlo,logN,logNhi), Ndetlim=detlim,
                 Wr=Wr, Wre=Wre, zp1=zp1,
                 ngoodpix=good.sum(), Nmult=Nmult, saturated=saturated)

