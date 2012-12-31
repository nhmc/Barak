r""" Tools for calculating dust attenuation.

**How dust attentuation is expressed in this module**

If :math:`I_\lambda` is the observed attenuated intensity of an
object, and :math:`I_{\lambda,0}` is the unattenuated intensity
modulated by an optical depth :math:`\tau_\lambda` due to dust
particles, then:

.. math::

  I_\lambda = I_{\lambda,0}\ e^{-\tau_\lambda}

Generally the attenuation is given in magnitudes in a band or at a
wavelength. For example, A(V) refers to the extinction in magnitudes
in the V band, and

.. math::

  E(B - V) \equiv A(B) - A(V)

is the difference in extinction between the B and V
bands. Empirically, dust attenuation in different parts of the Milky
Way's ISM can be described by single function of wavelength
parametrised by a normalisation A(V) and slope E(B - V). Another
commonly used quantity is

.. math::

  R(V) \equiv A(V) / E(B - V)

Analytic approximations for dust attenuation curves are often
calculated as a function of R(V), and then normalised by A(V) or, more
commonly, E(B - V).


The functions in his module give the attenuation for the Milky Way,
SMC and LMC, as a function of wavelength, and require that R(V) and
E(B-V) are given. The attenuation is given as the optical depth,
:math:`\tau(\lambda)`. This is related to :math:`A(\lambda)` in the
following way:

.. math::

  \tau(\lambda) = A(\lambda) / (2.5 \log_{10}(e))

**References**

- 'Astrophysics of Dust in Cold Clouds' by B.T. Draine:
  http://arxiv.org/abs/astro-ph/0304488
- 'Interstellar Dust Grains' by B.T. Draine:
  http://arxiv.org/abs/astro-ph/0304489

Note that much of the code in this module is inspired by Erik
Tollerud's `Astropysics <https://github.com/eteq/astropysics>`_.
"""

from utilities import get_data_path
import numpy as np

import warnings

datapath = get_data_path()
PATH_EXTINCT = datapath + '/dust_extinction/'

def starburst_Calzetti00(wa, EBmV, Rv=4.05):
    """ Dust extinction in starburst galaxies using the Calzetti
    relation.

    Find the extinction as a function of wavelength for the given
    E(B-V) using the relation from Calzetti et al.  R_v' = 4.05 is
    assumed (see equation (5) of Calzetti et al. 2000 ApJ, 533, 682)
    E(B-V) is the extinction in the stellar continuum.

    The wavelength array wa must be in Angstroms and sorted from
    low to high values.

    Parameters
    ----------
    wa : array_like
      Array of wavelengths in Angstroms at which to calculate the
      extinction.
    EBmV : float
      The E(B-V) value, a measure of the normalisation of the
      dust extinction curve.

    Returns
    -------
    tau : ndarray of floats, shape (N,)
      tau at each input wavelength.

    Examples
    --------
    wa = np.arange(1500, 3300, 0.1)
    tau = starburst_Calzetti00(wa, 0.08)

    # Assume a power law for the input flux
    flux = (wa/1500) ** -1.5
    extincted_flux = flux * np.exp(-tau)
    """

    wa = np.atleast_1d(wa)

    assert wa[0] < 22000 and wa[-1] > 1200

    # Note that EBmV is assumed to be Es as in equations (2) - (5)
    
    # Constants below assume wavelength is in microns.
    
    uwa = np.array(wa / 10000.)
    k = np.zeros_like(wa)

    def kshort(wa):
        return 2.659*(-2.156 + (1.509 - 0.198/wa + 0.011/wa**2)/wa) + Rv

    def klong(wa):
        return 2.659*(-1.857 + 1.040/wa) + Rv

    # populate k in a piece-wise fashion, extrapolating
    # below 1200 and above 22000 Angstroms
    if uwa[0] < 0.12:
        slope = (kshort(0.11) - kshort(0.12)) / (0.11 - 0.12)
        intercept = kshort(0.12) - slope * 0.12
        i = uwa.searchsorted(0.12)
        k[:i] = slope * uwa[:i] + intercept
    if uwa[0] < 0.63 and uwa[-1] > 0.12:
        i,j = uwa.searchsorted([0.12, 0.63])
        k[i:j] = kshort(uwa[i:j])
    if uwa[0] < 2.2 and uwa[-1] > 0.63:
        i,j = uwa.searchsorted([0.63, 2.2])
        k[i:j] = klong(uwa[i:j])
    if uwa[-1] >= 2.2:
        slope = (klong(2.19) - klong(2.2)) / (2.19 - 2.2)
        intercept = klong(2.19) - slope * 2.19
        i = uwa.searchsorted(2.2)
        k[i:] = slope * uwa[i:] + intercept

    # Note there is a typo in Calzetti et al. 2000 equation (2), there
    # should be a minus sign in the exponent.
    tau = EBmV * k / (2.5 * np.log10(np.e))
    if len(tau) == 1:
        return tau[0]
    else:
        return tau

def MW_Cardelli89(wa, EBmV=0.323, Rv=3.1):
    """ Milky Way Extinction law from Cardelli et
    al. 1989. (http://adsabs.harvard.edu/abs/1989ApJ...345..245C)

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms
    EBmV : float (default 0.323)
      E(B-V) value. This and R(V) set the slope and normalisation of
      the extinction.
    Rv : float (default 3.1)
      R(V). The default is for the diffues ISM, `Rv` of 5 is generally
      used for dense molecular clouds.  The default values of `Rv` and
      `EBmV` roughly correspond to A(V) = 1.

    Returns
    -------
    tau : ndarray or float if `wa` scalar
      The optical depth at each wavelength

    Note
    ----
    A power law extrapolation is used if there are any wavelengths
    past the IR or far UV limits of the Cardelli Law.
    """

    Av = Rv * EBmV

    wa = np.array(wa, ndmin=1, copy=False)

    # these correspond to x < 0.3, x > 10
    if (wa > 33333).any() or (1000 < wa).any():
        warnings.warn(
            'Some wavelengths outside CCM 89 extinction curve range, '
            'extrapolating')

    # CCM x is 1/microns
    x = 1e4 / wa 

    a = np.ones_like(x)
    b = np.ones_like(x)

    ir = (0.3 <= x) & (x <= 1.1)
    vis = (1.1 <= x) & (x <= 3.3)
    nuv1 = (3.3 <= x) & (x <= 5.9)
    nuv2 = (5.9 <= x) & (x <= 8)
    fuv = (8 <= x) & (x <= 10)

    # Infrared
    if ir.any():
        temp = x[ir]**1.61
        a[ir] = 0.574 * temp
        b[ir] = -0.527 * temp
    
    # NIR/optical
    if vis.any():
        co1 = (0.32999, -0.7753, 0.01979, 0.72085, -0.02427,
               -0.50447, 0.17699, 1.)
        a[vis] = np.polyval(co1, x[vis] - 1.82)
        co2 = (-2.09002, 5.3026, -0.62251, -5.38434, 1.07233,
               2.28305, 1.41338, 0.)
        b[vis] = np.polyval(co2, x[vis] - 1.82)
    
    # NUV
    if nuv1.any():
        a[nuv1] = 1.752 - 0.316*x[nuv1] - 0.104/((x[nuv1] - 4.67)**2 + 0.341)
        b[nuv1] = -3.09 + 1.825*x[nuv1] + 1.206/((x[nuv1] - 4.62)**2 + 0.263)

    if nuv2.any():
        y = x[nuv2] - 5.9
        Fa = -0.04473 * y**2 - 0.009779 * y**3
        Fb =  0.2130 * y**2 + 0.1207 * y**3
        a[nuv2] = 1.752 - 0.316*x[nuv2] \
                  - 0.104/((x[nuv2] - 4.67)**2 + 0.341) + Fa
        b[nuv2] = -3.09 + 1.825*x[nuv2] \
                  + 1.206/((x[nuv2] - 4.62)**2 + 0.263) + Fb
    
    # FUV
    if fuv.any():
        a[fuv] = np.polyval((-0.070,  0.137, -0.628, -1.073), x[fuv] - 8)
        b[fuv] = np.polyval(( 0.374, -0.42,   4.257,  13.67), x[fuv] - 8)

    Awa_on_Av = a + b / float(Rv)

    tau = Av * Awa_on_Av / (2.5 * np.log10(np.e))

    # extrapolate in log space (i.e. a power law) if there are any
    # wavelengths outside the Cardelli range.
    ir_extrap = x < 0.3   
    if ir_extrap.any():
        coeff = np.polyfit(np.log(x[ir][-2:]), np.log(tau[ir][-2:]), 1)
        tau[ir_extrap] = np.exp(np.polyval(coeff, np.log(x[ir_extrap])))

    uv_extrap = x > 10    
    if uv_extrap.any():
        coeff = np.polyfit(np.log(x[fuv][:2]), np.log(tau[fuv][:2]), 1)
        tau[uv_extrap] = np.exp(np.polyval(coeff, np.log(x[uv_extrap])))

    if len(x) == 1:
        return tau[0]
    else:
        return tau

def _FM(wa, c1, c2, c3, c4, gamma, x0, Rv):
    """ Base function for Extinction curves that use the form from
    Fitzpatrick & Massa 90.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.
    c1, c2, c3, c4, gamma, x0, Rv : float
      Constants that define the extinction law.

    Returns
    -------
    extinction : ndarray or float if `wa` is scalar
       extinction in the form E(lambda - V) / E(B - V)
    """

    x = 1e4 / np.array(wa, ndmin=1)

    xsq = x*x
    D = xsq * ((xsq - x0*x0)**2 + xsq*gamma*gamma)**-2
    FMf = c1 + c2*x + c3*D
    
    # Note EBmV is the normalization and is multiplied in at the end
    if len(x) == 1:
        if x >= 5.9:
            FMf += c4*(0.5392*(x-5.9)**2 + 0.05644*(x-5.9)**3)
        extinction = FMf[0] + Rv
    else:
        c4m = x >= 5.9
        FMf[c4m] += c4*(0.5392*(x[c4m] - 5.9)**2 + 0.05644*(x[c4m] - 5.9)**3)
        extinction = FMf + Rv

    return extinction

def LMC_Gordon03(wa, EBmV=0.3, Rv=3.41):
    """ LMC Extinction law from Gordon et al. 2003 LMC Average Sample.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.
    EBmV : float (default 0.3)
      E(B-V) value. This and R(V) set the slope and normalisation of
      the extinction.
    Rv : float (default 3.1)
      R(V).

    Returns
    -------
    tau : ndarray or float if `wa` scalar
      The optical depth at each wavelength.
    """
    c1,c2,c3,c4 = -0.890, 0.998, 2.719, 0.400
    x0, gamma = 4.579, 0.934
    temp = _FM(wa, c1, c2, c3, c4, x0, gamma, Rv)
    Av = Rv * EBmV
    tau = (temp * EBmV + Av) / (2.5 * np.log10(np.e))
    
    if len(tau) == 1:
        return tau[0]
    else:
        return tau
    
def SMC_Gordon03(wa, EBmV=0.2, Rv=2.74):
    """ SMC Extinction law from Gordon et al. 2003 SMC Bar Sample.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.
    EBmV : float (default 0.2)
      E(B-V) value. This and R(V) set the slope and normalisation of
      the extinction.
    Rv : float (default 2.74)
      R(V).

    Returns
    -------
    tau : ndarray or float if `wa` scalar
      The optical depth at each wavelength.
    """
    c1,c2,c3,c4 = -4.959, 2.264, 0.389, 0.461
    x0, gamma = 4.6, 1.0
    temp = _FM(wa, c1, c2, c3, c4, x0, gamma, Rv)
    Av = Rv * EBmV
    tau = (temp * EBmV + Av) / (2.5 * np.log10(np.e))    

    if len(tau) == 1:
        return tau[0]
    else:
        return tau
