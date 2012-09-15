""" Statistics-related functions.
"""
import numpy as np

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
    return poisson_min_max_limits(conf + 0.5*(100-conf), nevents)

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

    returns units of erg/s/cm^2/Hz/steradian
    """
    from constants import hplanck, c, kboltz
    return 2*h*nu**3 / (c**2 * (np.exp(hplanck*nu / (kboltz*T)) - 1))

def blackbody_lam(lam, T):
    """ Blackbody as a function of wavelength (cm) and temperature (K).

    returns units of erg/s/cm^2/cm/steradian
    """
    from constants import hplanck, c, kboltz
    return 2*h*c**2 / (lam**5 * (np.exp(hplanck*c / (lam*kboltz*T)) - 1))

