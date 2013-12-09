""" Statistics-related functions.
"""

# p2.6+ compatibility
from __future__ import division, print_function, unicode_literals
try:
    unicode
except NameError:
    unicode = basestring = str
    xrange = range

import numpy as np
from .utilities import between
import astropy.units as u

def _bisect(func, target, xlo=-10, xhi=10):
    """ Find x value such that func(x) = target.

    Assumes the value is positive and does log bisection.
    """
    if np.isnan([target, xlo, xhi]).any():
        return np.nan, np.nan
    target, xlo, xhi = (float(val) for val in (target, xlo, xhi))

    fmin = lambda x: func(10**x) - target

    dlo = fmin(xlo)
    dhi = fmin(xhi)
    if not np.sign(dlo * dhi) < 0:
        raise ValueError('Min and max x values do not bracket the target')

    if dhi < dlo:
        fmin = lambda x: - (func(10**x) - target)

    n = 0
    while True:
        x = 0.5 * (xlo + xhi)
        diff = fmin(x)
        if abs(diff) < 1e-6:
            break
        if n == 1000:
            raise RuntimeError('Max evaluations reached (1000)')
        elif diff > 0:
            xhi = x
        else:
            xlo = x
        n += 1

    return 10**x

def poisson_min_max_limits(conf, nevents):
    """ Calculate the minimum and maximum mean Poisson value mu
    consistent with seeing nevents at a given confidence level.

    conf: float
      95%, 90%, 68.3% or similar.
    nevents: int
      The number of events observed.

    Returns
    -------
    mulo, mhi : floats
      Mean number of events such that >= observed number of events
      nevents occurs in fewer than conf% of cases (mulo), and mean
      number of events such that <= nevents occurs in fewer than conf%
      of cases (muhi)
    """
    from scipy.stats import poisson
    nevents = int(nevents)
    conf = float(conf)

    if np.isnan(conf) or np.isnan(nevents):
        return np.nan, np.nan
    target = 1 - conf/100.
    if nevents == 0:
        mulo = 0
    else:
        mulo = _bisect(lambda mu: 1 - poisson.cdf(nevents-1, mu), target)

    muhi  = _bisect(lambda mu: poisson.cdf(nevents, mu), target)

    return mulo, muhi

def poisson_confidence_interval(conf, nevents):
    """ Find the Poisson confidence interval.

    Parameters
    ----------
    conf: float
      Confidence level in percent (95, 90, 68.3% or similar).
    nevents: int
      The number of events observed. If 0, then return the
      1-sided upper limit.

    Returns
    -------
    mulo, mhi : floats
      The two-sided confidence interval such that for mulo, >=
      observed number of events occurs in fewer than conf% of cases,
      and for muhi, <= nevents occurs in fewer than conf% of cases.
    """
    if nevents == 0:
        return poisson_min_max_limits(conf, nevents)
    return poisson_min_max_limits(conf + 0.5*(100 - conf), nevents)

def binomial_min_max_limits(conf, ntrial, nsuccess):
    """ Calculate the minimum and maximum binomial probability
    consistent with seeing nsuccess from ntrial at a given confidence level.

    conf: float
      95%, 90%, 68.3% or similar.
    ntrial, nsuccess: int
      The number of trials and successes.

    Returns
    -------
    plo, phi : floats
      Mean number of events such that >= observed number of events
      nevents occurs in fewer than conf% of cases (mulo), and mean
      number of events such that <= nevents occurs in fewer than conf%
      of cases (muhi)
    """
    from scipy.stats import binom
    nsuccess = int(nsuccess)
    ntrial = int(ntrial)
    conf = float(conf)

    if np.isnan(conf):
        return np.nan, np.nan
    target = 1 - conf/100.
    if nsuccess == 0:
        plo = 0
    else:
        plo = _bisect(lambda p: 1 - binom.cdf(nsuccess-1, ntrial, p), target,
                     xhi=0)
    if nsuccess == ntrial:
        phi = 1
    else:
        phi = _bisect(lambda p: binom.cdf(nsuccess, ntrial, p), target,
                     xhi=0)

    return plo, phi

def binomial_confidence_interval(conf, ntrial, nsuccess):
    """ Find the binomial confidence level.

    Parameters
    ----------
    conf: float
      Confidence level in percent (95, 90, 68.3% or similar).
    ntrial: int
      The number of trials.
    nsuccess: int
      The number of successes from the trials. If 0, then return the
      1-sided upper limit.

    Returns
    -------
    plo, phi : floats
      The two-sided confidence interval: probabilities such that >=
      observed number of successes occurs in fewer than conf% of cases
      (plo), and prob such that <= number of success occurs in fewer
      than conf% of cases (phi).
    """
    if nsuccess == 0:
        return binomial_min_max_limits(conf, ntrial, nsuccess)
    return binomial_min_max_limits(conf + 0.5*(100 - conf), ntrial, nsuccess)

def blackbody_nu(nu, T):
    """ Blackbody as a function of frequency (Hz) and temperature (K).

    Parameters
    ----------
    nu : array_like
      Frequency in Hz.

    Returns
    -------
    Jnu : ndarray
      Intensity with units of erg/s/cm^2/Hz/steradian

    See Also
    --------
    blackbody_lam
    """
    from .constants import hplanck, c, kboltz
    return 2*hplanck*nu**3 / (c**2 * (np.exp(hplanck*nu / (kboltz*T)) - 1))

def blackbody_lam(lam, T):
    """ Blackbody as a function of wavelength (Angstroms) and temperature (K).

    Parameters
    ----------
    lam : array_like
      Wavelength in Angstroms.

    Returns
    -------
    Jlam : ndarray
       Intensity with units erg/s/cm^2/Ang/steradian

    See Also
    --------
    blackbody_nu
    """
    from .constants import hplanck, c, kboltz
    # to cm
    lam = lam * 1e-8
    # erg/s/cm^2/cm/sr
    Jlam = 2*hplanck*c**2 / (lam**5 * (np.exp(hplanck*c / (lam*kboltz*T)) - 1))

    # erg/s/cm^2/Ang/sr
    return Jlam * 1e8

def remove_outliers(data, nsig_lo, nsig_hi, method='median',
                    nitermax=100, maxfrac_remove=0.95, verbose=False):
    """Strip outliers from a dataset, iterating until converged.

    Parameters
    ----------
    data : ndarray.
      data from which to remove outliers.
    nsig_lo : float
      Clip points that are this number of standard deviations below the
      centre of the data.
    nsig_hi : float
      Clip points that are this number of standard deviations above the
      centre of the data.
    method : {'mean', 'median', function}
      Method to find the central value.
    nitermax : int
      number of iterations before exit; defaults to 100
    maxfrac_remove : float (0.95)
      Clip at most this fraction of the original points.

    Returns
    -------
    mask : boolean array same shape as data
      This is False whereever there is an outlier.
    """

    data = np.asarray(data).ravel()

    funcs = {'mean': np.mean, 'median': np.median}
    if method in funcs:
        method = funcs[method]

    good = ~np.isnan(data)
    ngood = good.sum()
    niter = 0

    minpoints = int(len(data) * (1 - maxfrac_remove)) + 1
    while ngood > 0:
        d = data[good]
        centre = method(d)
        stdev = d.std()
        if stdev > 0:
            c0 = d > (centre - nsig_lo * stdev)
            c0 &= d < (centre + nsig_hi * stdev)
            good[good] = c0

        niter += 1
        ngoodnew = good.sum()
        if ngoodnew == ngood or niter > nitermax or ngoodnew <= minpoints:
            break
        if verbose:
            print(i, ngood, ngoodnew)
            print("centre, std", centre, stdev)

        ngood = ngoodnew

    return good


def find_conf_levels(a, pvals=[0.683, 0.955, 0.997]):
    """ Find the threshold value in an array that give confidence
    levels for an array of probabilities.

    Parameters
    ----------
    a : ndarray
      Array of probabilities. Can have more than one dimension.
    pvals : list of floats, shape (N,)
      Confidence levels to find the values for (must be between 0 and
      1). Default correspond to 1, 2 and 3 sigma. Must be smallest to
      largest.

    Returns
    -------
    athresh : list of shape (N,)
      The threshold values of `a` giving the confidence ranges in
      pvals.
    """
    assert all(0 <= float(p) <= 1 for p in pvals)
    assert np.allclose(np.sort(pvals), pvals)
    a = np.asarray(a).ravel()

    assert not np.isnan(a).any()
    assert not np.isinf(a).any()

    tot = a.sum()
    conflevels = [p * tot for p in pvals]
    asorted = np.sort(a)

    out = []
    i = -2
    for iconf, clevel in enumerate(conflevels):
        while asorted[i:].sum() < clevel:
            i -= 1
            if i == -len(a):
                i += 1
                break
        #print (i, asorted[i], asorted[i:].sum(), clevel, pvals[iconf])
        out.append(asorted[i])

    return out


def Schechter_Mag(M):
    """ phi = phi_star * (M / Mstar)**alpha * np.exp(M/Mstar)

    From Reddy et al. 2008?
    Table 7

    z=1.9 to 2.7

    alpha         M*           1e3* phi*

    # ground
    -1.88 0.27   -21.01 0.38   1.62  0.46
    # fixed alpha
    -1.60 0.00   -20.60 0.08   3.31  0.22

    z=2.7 to 3.4

    # ground
    -1.85 0.26   -21.12 0.02   1.12  0.52
    # ground + space
    -1.57 0.11   -20.84 0.12   1.66  0.63
    """

    MSTAR = -21.01
    ALPHA = -1.88
    # this is in h_0.7^3 * Mpc^-3 * mag^-1
    PHI_STAR = 1.62e-3
    Mterm = 10**(-0.4*(M - MSTAR))
    return 0.4 * np.log(10) * PHI_STAR * Mterm**(ALPHA + 1) * np.exp(-Mterm)


def BX_number_density(lmin, lmax=10, Mstar=-21.0):
    """ Give the low luminosity cutoff for the

    lmin and lmax are the luminosity integration limits in units of L*.

    Returns density in Mpc^-3 h(0.7)^3
    """
    Mlo = Mstar - np.log10(lmax) * 2.5
    Mhi = Mstar - np.log10(lmin) * 2.5
    print('using Mlo=%.2f, Mhi=%.2f' % (Mlo, Mhi))
    from scipy.integrate import quad
    return quad(Schechter_Mag, Mlo, Mhi)[0]

def slit_losses(seeing_on_slitwidth):
    """Find the slit losses for a given seeing and slit width.

    Parameters
    ----------
    seeing_on_slitwidth : array_like
       The ratio of seeing FWHM to slit width.

    Returns
    -------
    losses: float
      Fraction of light that falls outside the slit.

    Notes
    -----
    Assumes a 2d Gaussian for the seeing profile.
    """
    # derived from this Sage calculation:

    # var('x y w')
    # assume(w > 0)
    # # assume seeing FWHM is always 1
    # sig = 1 / (2*sqrt(2*log(2)))
    # expr = exp(-(x/sig)^2/2 - (y/sig)^2/2
    # # Note that minus sign below is needed due a bug in Sage, I think.
    # area = -integral(integral(expr), x, -w/2, w/2), y, -oo, oo)
    # total = integral(integral(expr), x, -oo, oo), y, -oo, oo)
    # ans = 1 - area/total

    from math import sqrt, log, erf
    from .utilities import iscontainer
    const = sqrt(log(2))
    if iscontainer(seeing_on_slitwidth):
        out = np.array([erf(const / r) for r in seeing_on_slitwidth])
        return out
    return 1 - erf(const / seeing_on_slitwidth)


def polyfitr(x, y, order=2, clip=6, xlim=None, ylim=None, mask=None,
             debug=False):
    """ Fit a polynomial to data, rejecting outliers.

    Fits a polynomial f(x) to data, x,y.  Finds standard deviation of
    y - f(x) and removes points that differ from f(x) by more than
    clip*stddev, then refits.  This repeats until no points are
    removed.

    Inputs
    ------
    x,y:
        Data points to be fitted.  They must have the same length.
    order: int (2)
        Order of polynomial to be fitted.
    clip: float (6)
        After each iteration data further than this many standard
        deviations away from the fit will be discarded.
    xlim: tuple of maximum and minimum x values, optional
        Data outside these x limits will not be used in the fit.
    ylim: tuple of maximum and minimum y values, optional
        As for xlim, but for y data.
    mask: sequence of pairs, optional
        A list of minimum and maximum x values (e.g. [(3, 4), (8, 9)])
        giving regions to be excluded from the fit.
    debug: boolean, default False
        If True, plots the fit at each iteration in matplotlib.

    Returns
    -------
    coeff, x, y:
        x, y are the data points contributing to the final fit. coeff
        gives the coefficients of the final polynomial fit (use
        np.polyval(coeff,x)).

    Examples
    --------
    >>> x = np.linspace(0,4)
    >>> np.random.seed(13)
    >>> y = x**2 + np.random.randn(50)
    >>> coeff, x1, y1 = polyfitr(x, y)
    >>> np.allclose(coeff, [1.05228393, -0.31855442, 0.4957111])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, order=1, xlim=(0.5,3.5), ylim=(1,10))
    >>> np.allclose(coeff, [3.23959627, -1.81635911])
    True
    >>> coeff, x1, y1 = polyfitr(x, y, mask=[(1, 2), (3, 3.5)])
    >>> np.allclose(coeff, [1.08044631, -0.37032771, 0.42847982])
    True
    """
    good = ~np.isnan(x) & ~np.isnan(y)
    x = np.asanyarray(x[good])
    y = np.asanyarray(y[good])
    isort = x.argsort()
    x, y = x[isort], y[isort]

    keep = np.ones(len(x), bool)
    if xlim is not None:
        keep &= (xlim[0] < x) & (x < xlim[1])
    if ylim is not None:
        keep &= (ylim[0] < y) & (y < ylim[1])
    if mask is not None:
        badpts = np.zeros(len(x), bool)
        for x0,x1 in mask:
            badpts |=  (x0 < x) & (x < x1)
        keep &= ~badpts

    x,y = x[keep], y[keep]
    if debug:
        fig = pl.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,y,'.')
        ax.set_autoscale_on(0)
        pl.show()

    coeff = np.polyfit(x, y, order)
    if debug:
        pts, = ax.plot(x, y, '.')
        poly, = ax.plot(x, np.polyval(coeff, x), lw=2)
        pl.show()
        raw_input('Enter to continue')
    norm = np.abs(y - np.polyval(coeff, x))
    stdev = np.std(norm)
    condition =  norm < clip * stdev
    y = y[condition]
    x = x[condition]
    while norm.max() > clip * stdev:
        if len(y) < order + 1:
            raise Exception('Too few points left to fit!')
        coeff = np.polyfit(x, y, order)
        if debug:
            pts.set_data(x, y)
            poly.set_data(x, np.polyval(coeff, x))
            pl.show()
            raw_input('Enter to continue')
        norm = np.abs(y - np.polyval(coeff, x))
        stdev = norm.std()
        condition =  norm < clip * stdev
        y = y[condition]
        x = x[condition]

    return coeff,x,y

class SFR_Lum:
    """ Class for converting from luminosities to a star
    formation rate and vice versa using the Kennicutt 98 relations.

    All the calculated values can be shown by using `print`. For
    example:

    >>> import astropy.units as u
    >>> sfr_rel = SFR_Lum(SFR=10 * u.M_sun / u.yr)
    >>> print sfr_rel

    Relations between Star formation rate and luminosities in
    different emission lines or broad bands from Kennicutt Jr. R. C.,
    1998, ARAA , 36, 189."""

    def __init__(self, L_Ha=None, L_Lya=None, L_OII=None, L_UV=None, L_FIR=None,
                 SFR=None):
        """ One luminosity or a SFR keyword argument must given.
        """

        const = u.M_sun / u.yr / (u.erg / u.s)

        if L_Lya is not None:
            """ Using Kennicutt 98 H-alpha relation, and converting to Ly-a
            assuming case B ratio or 8.7 (Brocklehurst et al. 1971)."""
            SFR = 9.1e-43 * const * L_Lya
        elif L_Ha is not None:
            SFR = 7.9e-42 * const * L_Ha
        elif L_OII is not None:
            SFR = 1.4e-41 * const * L_OII
        elif L_UV is not None:
            SFR = 1.4e-28 * u.M_sun / u.yr / (u.erg / u.s / u.Hz) * L_UV
        elif L_FIR is not None:
            SFR = 4.5e-44 * const * L_FIR
        elif SFR is None:
            raise KeyError('Must specify a SFR or luminosity!')

        self.const = const
        self.SFR = SFR.to(u.M_sun / u.yr)
        self.set_Lum()

    def __str__(self):
        s = []
        for attr in 'SFR L_Lya L_Ha L_OII L_UV L_FIR'.split():
            s.append('{0}:  {1:.2g}'.format(attr, getattr(self, attr)))
        return '\n'.join(s)

    def set_Lum(self):
        """ Set all the Luminosities from the SFR.
        """
        const = self.const

        self.L_Lya = (self.SFR / (9.1e-43 * const)).to(u.erg / u.s)
        self.L_Ha = (self.SFR / (7.9e-42 * const)).to(u.erg / u.s)
        self.L_OII = (self.SFR / (1.4e-41 * const)).to(u.erg / u.s)
        self.L_UV = (self.SFR / (1.4e-28 * u.M_sun / u.yr /
                                 (u.erg / u.s / u.Hz))).to(u.erg / u.s / u.Hz)
        self.L_FIR = (self.SFR / (4.5e-44 * const)).to(u.erg / u.s)

def Gaussian(x, x0, sigma, height):
    """ Gaussian.

    Convert between sigma and FWHM using the relation:

    FWHM = sigma * 2 * sqrt(2 * ln(2)) ~ sigma * 2.35482
    """
    return height * np.exp(-0.5 * ((x-x0)/sigma)**2)

def chi2_prob(chi2val, df):
    """ Calculate the probability of finding this chi2 value or higher/lower.

    Parameters
    ----------
    chi2val : float
      chi^2 value
    df :
      Number of degrees of freedom

    Returns
    -------
    phi,plo : floats
      The probability of finding this chi^2 value or higher and this
      chi^2 value or lower (plo = 1 - phi).

    Notes
    -----

    This uses the probability density function for the chi2
    distribution, which is:

      chi2.pdf(x,df) = 1 / (2*gamma(df/2)) * (x/2)**(df/2-1) * exp(-x/2)
    """
    from scipy.stats import chi2
    from scipy.integrate import quad
    phi, err = quad(chi2.pdf, chi2val, np.inf, (df,))
    return phi, 1-phi

def calc_chi2(data, data_sigma, model):
    """ Calculate the Chi^2 given data and errors and a model.

    Parameters
    ----------
    data : array_like, shape (N,)
      Data values.
    data_sigma : array_like, shape (N,)
      One sigma errors on each data value.

    Returns
    -------
    chi2 : float
      The sum of ((data - model) / data_sigma)**2
    """
    data,data_sigma,model = map(np.asarray, (data, data_sigma, model))
    chi = (data - model) / data_sigma
    return np.dot(chi, chi)
    
