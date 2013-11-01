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
calculated as a function of R(V), and then normalised by A(V) or
E(B - V).

The attenuation as a function of wavelength for a given extinction
curve is often given as as :math:`A(\lambda)/A(V)`. This can be
converted to an optical depth in the following way:

.. math::

  \tau(\lambda) = \frac{A(\lambda)}{A(V)}\ R(V)\  E(B-V) / (2.5 \log_{10}(e))

The functions in this module use an `ExtinctionCurve` class to
represent the attentuation as a function of wavelength for the Milky
Way, SMC and LMC, and a starburst galaxy. Using attributes of this
class, you can express the extinction curve as an optical depth
:math:`\tau`, as :math:`A(\lambda)/A(V)` or as
:math:`E(B-\lambda)/E(B-V)`. R(V), A(V) and E(B-V) are also stored as
attributes.

**References**

- 'Astrophysics of Dust in Cold Clouds' by B.T. Draine:
  http://arxiv.org/abs/astro-ph/0304488
- 'Interstellar Dust Grains' by B.T. Draine:
  http://arxiv.org/abs/astro-ph/0304489

Note that much of the code in this module is inspired by Erik
Tollerud's `Astropysics <https://github.com/eteq/astropysics>`_.
"""

# p2.6+ compatibility
from __future__ import division, print_function, unicode_literals

from .utilities import between
from .interp import interp_Akima
import numpy as np

import warnings

# interpolation limits
W0, W1 = 2850, 3700

class ExtinctionCurve(object):
    def __init__(self, wa, Rv, AlamAv,
                 name='ExtinctionCurve', EBmV=None):
        self._name = name
        self._wa = wa
        self._Rv = Rv
        self._AlamAv = AlamAv
        self._ElamV = None
        if EBmV is not None:
            self.set_EBmV(EBmV)
        else:
            self._EBmV = None
            self._Av = None
            self._tau = None

    def __repr__(self):
        s = '< {0._name}: R(V)={0._Rv}, E(B-V)={0._EBmV}, A(V)={0._Av} >'
        return s.format(self)

    @property
    def EBmV(self):
        return self._EBmV

    def set_EBmV(self, EBmV):
        """ Sets Av, tau and Alam in addition to EBmV
        """
        self._EBmV = EBmV
        self._Av = EBmV * self.Rv
        self._tau = tau_from_AlamAv(self._AlamAv, self._Av)

    @property
    def ElamV(self):
        if self._ElamV is None:
            self._ElamV = ElamV_from_AlamAv(self._AlamAv, self._Rv)
        return self._ElamV

    @property
    def wa(self):
        return self._wa

    @property
    def Rv(self):
        return self._Rv

    @property
    def AlamAv(self):
        return self._AlamAv

    @property
    def Av(self):
        return self._Av

    @property
    def tau(self):
        return self._tau

    @property
    def name(self):
        return self._name

def starburst_Calzetti00(wa, Rv=4.05, EBmV=None):
    """ Dust extinction in starburst galaxies using the Calzetti
    relation.

    Find the extinction as a function of wavelength for the given
    E(B-V) using the relation from Calzetti et al.  R_v' = 4.05 is
    assumed (see equation [5] of Calzetti et al. 2000 ApJ, 533, 682)
    E(B-V) is the extinction in the stellar continuum.

    The wavelength array wa must be in Angstroms and sorted from
    low to high values.

    Parameters
    ----------
    wa : array_like
      Array of wavelengths in Angstroms at which to calculate the
      extinction.
    Rv : float (default 4.05)
      Constant determining the slope of the extinction law.

    Returns
    -------
    Ext : ExtinctionCurve instance

    References
    ----------
    Calzetti et al. 2000 ApJ, 533, 682:

      http://adsabs.harvard.edu/abs/2000ApJ...533..682C

    Examples
    --------
    wa = np.arange(1500, 3300, 0.1)
    ext = starburst_Calzetti00(wa, 0.08)

    # Assume a power law for the input flux
    flux = (wa/1500) ** -1.5
    extincted_flux = flux * np.exp(-ext.tau)
    """

    wa = np.atleast_1d(wa)

    assert wa[0] < 22000 and wa[-1] > 1200

    # Note that EBmV is assumed to be Es as in equations (2) - (5)

    # k is A(lambda) / E(B - V)
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

    # Note there is a typo in Calzetti et al. 2000 equation (2), there
    # should be a minus sign in the exponent.
    ElamV = k - Rv
    AlamAv = AlamAv_from_ElamV(ElamV, Rv)

    # assume MW extinction law above these wavelengths. Note we can't
    # interpolate in E(lambda - V) / E(B - V) because this gives
    # unphysical values on converting to A(lambda).
    c0 = wa > W0
    if c0.any():
        c1 = between(wa, W0, W1)
        if c1.any():
            AlamAv[c0] = MW_Cardelli89(wa[c0], 3.1).AlamAv
            AlamAv[c1] = interp_Akima(wa[c1], wa[~c1], AlamAv[~c1])

    if len(wa) == 1:
        return ExtinctionCurve(wa[0], Rv, AlamAv[0], EBmV=EBmV,
                               name='starburst_Calzetti00')
    else:
        return ExtinctionCurve(wa, Rv, AlamAv, EBmV=EBmV,
                               name='starburst_Calzetti00')

def MW_Cardelli89(wa, EBmV=None, Rv=3.1):
    """ Milky Way Extinction law from Cardelli et al. 1989.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms
    Rv : float (default 3.1)
      R(V). The default is for the diffuse ISM, `Rv` of 5 is generally
      used for dense molecular clouds.

    Returns
    -------
    Ext : ExtinctionCurve instance

    Notes
    -----
    A power law extrapolation is used if there are any wavelengths
    past the IR or far UV limits of the Cardelli Law.

    References
    ----------
    http://adsabs.harvard.edu/abs/1989ApJ...345..245C
    """

    wa = np.array(wa, ndmin=1, copy=False)

    # these correspond to x < 0.3, x > 10
    #if (wa > 33333).any() or (1000 < wa).any():
    #    warnings.warn(
    #        'Some wavelengths outside CCM 89 extinction curve range, '
    #        'extrapolating')

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

    AlamAv = a + b / float(Rv)
    
    # extrapolate in log space (i.e. a power law) if there are any
    # wavelengths outside the Cardelli range.
    ir_extrap = x < 0.3   
    if ir_extrap.any():
        coeff = np.polyfit(np.log(x[ir][-2:]), np.log(AlamAv[ir][-2:]), 1)
        AlamAv[ir_extrap] = np.exp(np.polyval(coeff, np.log(x[ir_extrap])))

    uv_extrap = x > 10    
    if uv_extrap.any():
        coeff = np.polyfit(np.log(x[fuv][:2]), np.log(AlamAv[fuv][:2]), 1)
        AlamAv[uv_extrap] = np.exp(np.polyval(coeff, np.log(x[uv_extrap])))

    if len(x) == 1:
        return ExtinctionCurve(wa[0], Rv, AlamAv[0], EBmV=EBmV,
                               name='MW_Cardelli89')
    else:
        return ExtinctionCurve(wa, Rv, AlamAv, EBmV=EBmV,
                               name='MW_Cardelli89')


def ElamV_FM(wa, c1, c2, c3, c4, x0, gamma):
    """ Base function for Extinction curves that use the form from
    Fitzpatrick & Massa 90.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.
    c1, c2, c3, c4, gamma, x0 : float
      Constants that define the extinction law.

    Returns
    -------
    ElamV : ndarray or float if `wa` is scalar
       extinction in the form E(lambda - V) / E(B - V)
    """

    x = 1e4 / np.array(wa, ndmin=1)

    xsq = x*x
    D = xsq / ((xsq - x0*x0)**2 + xsq*gamma*gamma)
    FMf = c1 + c2*x + c3*D

    if len(x) == 1:
        if x >= 5.9:
            FMf += c4*(0.5392*(x - 5.9)**2 + 0.05644*(x - 5.9)**3)
        ElamV = FMf[0]
    else:
        c4m = x >= 5.9
        FMf[c4m] += c4*(0.5392*(x[c4m] - 5.9)**2 + 0.05644*(x[c4m] - 5.9)**3)
        ElamV = FMf

    return ElamV

def LMC_Gordon03(wa, EBmV=None):
    """ LMC Extinction law from Gordon et al. 2003 LMC Average Sample.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.

    Returns
    -------
    Ext : ExtinctionCurve instance
      A(lambda)/A(V) at each wavelength and Rv for the
      LMC Average sample.

    References
    ----------
    http://adsabs.harvard.edu/abs/2003ApJ...594..279G

    Notes
    -----
    At far IR wavelengths, a MW extinction curve is assumed. 
     """
    
    wa = np.array(wa, copy=False, ndmin=1)
    c1,c2,c3,c4 = -0.890, 0.998, 2.719, 0.400
    x0, gamma = 4.579, 0.934
    ElamV = ElamV_FM(wa, c1, c2, c3, c4, x0, gamma)
    AlamAv = AlamAv_from_ElamV(ElamV, 3.41)
    # assume MW extinction law above these wavelengths
    c0 = wa > W0
    if c0.any():
        c1 = between(wa, W0, W1)
        if c1.any():
            AlamAv[c0] = MW_Cardelli89(wa[c0], 3.1).AlamAv
            AlamAv[c1] = interp_Akima(wa[c1], wa[~c1], AlamAv[~c1])
        
    if len(AlamAv) == 1:
        return ExtinctionCurve(wa[0], 3.41, AlamAv[0], EBmV=EBmV,
                               name='LMC_Gordon03')
    else:
        return ExtinctionCurve(wa, 3.41, AlamAv, EBmV=EBmV,
                               name='LMC_Gordon03')

def LMC2_Gordon03(wa, EBmV=None):
    """ LMC Extinction law from Gordon et al. 2003 LMC supershell
    sample.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.

    Returns
    -------
    Ext : ExtinctionCurve instance
      A(lambda)/A(V) at each wavelength and Rv for the
      LMC supershell sample.

    References
    ----------
    http://adsabs.harvard.edu/abs/2003ApJ...594..279G    

    Notes
    -----
    At far IR wavelengths, a MW extinction curve is assumed. 
     """
    c1,c2,c3,c4 = -1.475, 1.132, 1.463, 0.294
    x0, gamma = 4.558, 0.945
    ElamV = ElamV_FM(wa, c1, c2, c3, c4, x0, gamma)
    # assume MW extinction law above these wavelengths
    AlamAv = AlamAv_from_ElamV(ElamV, 2.76)
    c0 = wa > W0
    if c0.any():
        c1 = between(wa, W0, W1)
        if c1.any():
            AlamAv[c0] = MW_Cardelli89(wa[c0], 3.1).AlamAv
            AlamAv[c1] = interp_Akima(wa[c1], wa[~c1], AlamAv[~c1])

    if len(AlamAv) == 1:
        return ExtinctionCurve(wa[0], 2.76, AlamAv[0], EBmV=EBmV,
                               name='LMC2_Gordon03')
    else:
        return ExtinctionCurve(wa, 2.76, AlamAv, EBmV=EBmV,
                               name='LMC2_Gordon03')

    
def SMC_Gordon03(wa, EBmV=None):
    """ SMC Extinction law from Gordon et al. 2003 SMC Bar Sample.

    Parameters
    ----------
    wa : array_like
      One or more wavelengths in Angstroms.
      
    Returns
    -------
    Ext : ExtinctionCurve instance
      A(lambda)/A(V) at each wavelength and Rv for the
      SMC bar sample.

    References
    ----------
    http://adsabs.harvard.edu/abs/2003ApJ...594..279G

    Notes
    -----
    At far IR wavelengths, a MW extinction curve is assumed. 
    """
    c1,c2,c3,c4 = -4.959, 2.264, 0.389, 0.461
    x0, gamma = 4.6, 1.0
    ElamV = ElamV_FM(wa, c1, c2, c3, c4, x0, gamma)
    AlamAv = AlamAv_from_ElamV(ElamV, 2.74)
    # assume MW extinction law above these wavelengths
    c0 = wa > W0
    if c0.any():
        c1 = between(wa, W0, W1)
        if c1.any():
            AlamAv[c0] = MW_Cardelli89(wa[c0], 3.1).AlamAv
            AlamAv[c1] = interp_Akima(wa[c1], wa[~c1], AlamAv[~c1])

    if len(AlamAv) == 1:
        return ExtinctionCurve(wa[0], 2.74, AlamAv[0], EBmV=EBmV,
                               name='SMC_Gordon03')
    else:
        return ExtinctionCurve(wa, 2.74, AlamAv, EBmV=EBmV,
                               name='SMC_Gordon03')

def tau_from_AlamAv(AlamAv, Av):
    """ Find tau(lambda) from A(lambda)/A(V)

    Note that A(V) = E(B - V) * R(V)
    """
    return AlamAv * Av / (2.5 * np.log10(np.e))

def AlamAv_from_ElamV(ElamV, Rv):
    """ Find A(lambda)/A(V) from E(lambda - V) / E(B - V).
    """
    return (ElamV + Rv) / Rv

def ElamV_from_AlamAv(AlamAv, Rv):
    """ Find E(lambda - V) / E(B - V) from A(lambda)/A(V).
    """
    return AlamAv*Rv - Rv
1
