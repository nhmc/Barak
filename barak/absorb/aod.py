""" Calculate optical depths and colun densities using the apparent
optical depth method.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from math import pi, isnan, isinf, log

import astropy.constants as C
import astropy.units as u

try:
    xrange
except NameError:
    xrange = range

c_kms = C.c.to(u.km / u.s).value
c_cms = C.c.to(u.cm / u.s).value
# this constant gets used in several functions
e2_me_c = (C.e.gauss**2 / (C.m_e*C.c)).to(u.cm**2 / u.s).value

__all__ = ['Nvel_from_tau', 'Nlam_from_tau', 'tau_from_nfl_ner',
           'tau_AOD', 'calc_N_AOD']

def Nvel_from_tau(tau, wa, osc):
    """ Find the column density per velocity interval.

    Parameters
    ----------
    tau : array_like, shape (N,)
      Array of optical depths/
    wa : float or astropy quantity
      Rest wavelength of the transition. 
    osc : float
      Transition oscillator strength.

    Returns
    -------
    Nvel : ndarray, shape (N,)
      The column density per velocity interval, with units cm^-2
      (km/s)^-1 Multiply this by a velocity interval in km/s to get a
      column density.
    """
    # the 1e5 here converts from (cm/s)^-1 to (km/s)^-1
    return tau / (pi * e2_me_c * osc) / (wa * 1e-8) * 1e5


def Nlam_from_tau(tau, wa, osc):
    """ Find the column density per rest wavelength interval.

    Parameters
    ----------
    tau : array_like, shape (N,)
      Array of optical depths/
    wa : float
      Rest wavelength of the transition.
    osc : float
      Transition oscillator strength.

    Returns
    -------
    Nlam : ndarray, shape (N,)
      The column density per wavelength interval, with units cm^-2
      Angstrom^-1. Multiply this by the rest wavelength interval in
      Angstroms corresponding to the width of one input pixel to get a
      column density.
    """
    # the 1e-8 here converts from cm^-1 to Angstrom^-1
    return tau / (pi * e2_me_c * osc) * c_cms / (wa * 1e-8)**2 * 1e-8


def tau_from_nfl_ner(nfl, ner, sf=1):
    """ Find the optical depth given a normalised flux and error.

    Parameters
    ----------
    nfl, ner : array_like or float
      Normalised fluxes and 1 sigma errors.

    sf : array_like or float (optional)
      Multiplier to give the optical depth. Must either be scalar, or have
      the same length as nfl and ner.

    Returns
    -------
    tau : ndarray or float
      optical depth

    Notes
    -----
    To calculate upper limits, it's better to use the optically thin
    approximation via the equivalent width calculation in calc_Wr().

    For saturated lines, this estimate is a lower limit. The scale
    factor is typically 1, it can be non zero if you want to scale the
    optical depth of one transition to another (e.g. to infer CIV 1550
    from CIV 1558).
    """
    # fast path to avoid expensive checks for array inputs
    assert ner > 0
    try:
        float(nfl)
    except TypeError:
        # slow path to deal with array inputs
        nfl = np.atleast_1d(nfl)
        ner = np.atleast_1d(ner)
        sf = np.atleast_1d(sf)
        if len(sf) == 1:
            sf = np.ones(len(nfl), float) * sf[0]
        tau = np.empty(len(nfl), float)
        c0 = nfl > 1
        tau[c0] = 0
        c1 = nfl < ner
        tau[c1]  = -np.log(ner[c1]) * sf[c1]
        c2 = ~(c0 | c1)
        tau[c2] = -np.log(nfl[c2]) * sf[c2]
        return tau

    if nfl >= 1:
        return 0
    elif nfl < ner:
        # then lower limit
        return -log(ner) * sf
    else:
        try:
            return -log(nfl) * sf
        except ValueError:
            import pdb; pdb.set_trace()



def _tau_cont_mult(nfl, ner, colo_mult, cohi_mult,
                  zerolo_nsig, zerohi_nsig, sf=1):
    """ find the optical depth from a flux and error taking into
    account a multiplicative uncertainty in the continuum.

    See find_tau_from_nfl_ner() for more details.

    Parameters
    ----------
    nfl, ner : floats
      Normalised fluxes and 1 sigma errors.

    colo_mult, cohi_mult : float
      Multiplier to the continuum to represent the continuum
      uncertainty. For example colo_mult=0.97, cohi_mult=1.03
      calculates tau for a continuum 3% lower and 3% higher than
      actual continuum.

    zerolo_nsig, zerohi_nsig : float
      Zero level offsets in units of 1 sigma (both > 0).

    sf : float
      Multiplier to give the optical depth.

    Returns
    -------
    taulo, tau, tauhi, nfl_min, nfl_max :
      Minimum, best and maximum optical depths, and the flux minimum
      and maximum based on the input continuum scale factors.
    """
    # highest flux from continuum, zero level, and 1 sigma variation
    zoff = zerolo_nsig * ner
    nfl_max = max(nfl / colo_mult, nfl + ner, (nfl + zoff) / (1 + zoff))
    taulo = tau_from_nfl_ner(nfl_max, ner / colo_mult, sf=sf)
    # lowest flux from continuum, zero_level, and 1 sigma variation, but no lower
    # than ner
    zoff = zerohi_nsig * ner
    nfl_min = max(
        min(nfl / cohi_mult, nfl - ner, (nfl - zoff) / (1 - zoff)),
        ner)
    tauhi = tau_from_nfl_ner(nfl_min, ner / cohi_mult, sf=sf)
    tau = tau_from_nfl_ner(nfl, ner, sf=sf)

    return taulo, tau, tauhi, nfl_min, nfl_max



def tau_AOD(nfl, ner, colo_nsig=0.5, cohi_nsig=0.5,
            colo=None, cohi=None,
            zerolo_nsig=0.5, zerohi_nsig=0.5, sf=1):
    """ Find the optical depth from a flux and error using the
    apparent optical depth method.

    Parameters
    ----------
    nfl, ner : arrays shape N
      Normalised flux and 1 sigma error.
    colo_nsig, cohi_nsig : float (default 0.5)
      Continuum offsets in units of 1 sigma (both > 0)
    colo, cohi : float (default None)
      Continuum offsets as a fraction of continuum (e.g 0.03 = 3%). Override
      colo_nsig, cohi_nsig.      
    zerolo_nsig, zerohi_nsig : float (deafult 0.5)
      Zero level offsets in units of 1 sigma (both > 0).
    sf : float (default 1)
      Multiplier for the optical depth (useful to scale transitions
      with different oscillator strengths to match each other.)

    Returns
    -------
    taulo, tau, tauhi, nfl_min, nfl_max :
      Minimum, best and maximum optical depths, and the flux minimum
      and maximum based on the input continuum scale factors.

    See find_tau_from_nfl_ner() for more details.
    """

    if cohi is not None:
        cohi_mult = 1 + cohi
    else:
        cohi_mult = 1 + ner * cohi_nsig

    if colo is not None:
        colo_mult = 1 - colo
    else:
        colo_mult = 1 - ner * colo_nsig

    return _tau_cont_mult(nfl, ner, colo_mult, cohi_mult,
                         zerolo_nsig, zerohi_nsig, sf=sf)


def calc_N_AOD(wa, nfl, ner, wa0, osc, redshift=None,
               colo_nsig=0.5, cohi_nsig=0.5,
               zerolo_nsig=0.5, zerohi_nsig=0.5,
               colo=None, cohi=None):
    """ Find the column density for a single transition using the AOD
    method.

    Parameters
    ----------
    wa, nfl, ner : arrays shape N
      wavelengths (Angstroms), normalised flux and 1 sigma error.
    wa0, osc : floats
      Transition rest wavelength and oscillator strength.
    redshift : float
      Transition redshift. Default is None, which means it is
      estimated from the wa array.
    colo_nsig, cohi_nsig : float (default 0.5)
      Continuum offsets in units of 1 sigma (both > 0)
    zerolo_nsig, zerohi_nsig : float (deafult 0.5)
      Zero level offsets in units of 1 sigma (both > 0).
    colo, cohi : float (default None)
      Continuum offsets as a fraction of continuum (e.g 0.03 =
      3%). These override colo_nsig, cohi_nsig.
    sf : float (default 1)
      Multiplier for the optical depth (useful to scale transitions
      with different oscillator strengths to match each other.)

    Returns
    -------
    taulo, tau, tauhi, sat :
      Minimum, best and maximum optical depths, and a flag showing
      whether the transition is saturated.
    """
    n = len(wa)
    assert len(nfl) == len(ner) == n

    if redshift is None:
        zp1 = 0.5*(wa[0] + wa[-1]) / wa0
    else:
        zp1 = redshift + 1

    taulo, tau, tauhi = [], [], []
    saturated = False
    nsat = 0
    for i in xrange(n):
        if not (ner[i] > 0) or isnan(nfl[i]) or isinf(nfl[i]) \
               or isnan(ner[i]):
            taulo.append(np.nan)
            tau.append(np.nan)
            tauhi.append(np.nan)
            continue

        tlo, t, thi, f0, f1 = tau_AOD(
            nfl[i], ner[i], colo_nsig=colo_nsig, cohi_nsig=cohi_nsig,
            zerolo_nsig=zerolo_nsig, zerohi_nsig=zerohi_nsig,
            colo=colo, cohi=cohi)
        if f0 <= 0.5*ner[i]:
            nsat += 1
        taulo.append(tlo)
        tau.append(t)
        tauhi.append(thi)

    # if half the pixels are below 0.5 * error, flag it as saturated.
    if nsat / float(n) > 0.5:
        saturated = True

    imid = n // 2
    dw0 = (wa[imid+1] - wa[imid]) / zp1

    logNvals = []
    for t in taulo, tau, tauhi:
        # interpolate across any bad values
        t = np.array(t)
        c0 = np.isnan(t)
        if c0.any():
            x = np.arange(len(t))
            t[c0] = np.interp(x[c0], x[~c0], t[~c0])
        Nlam = Nlam_from_tau(t, wa0, osc)
        logNvals.append(np.log10(np.sum(Nlam * dw0)))

    logNlo, logN, logNhi = logNvals

    return logNlo, logN, logNhi, saturated
