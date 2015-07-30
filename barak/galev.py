"""
Galaxy-evolution related utilities

"""
import astropy.units as u
import numpy as np
from scipy.integrate import quad

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

    def __init__(self, L_Ha=None, L_Lya=None, L_OII=None, L_UV=None,
                 L_FIR=None, SFR=None):
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


def cosmic_sfr(z):
    """ Cosmic star formation rate in Msun / yr / cubic Mpc.

    Equation 16 from Madau and Dickinson 2014.

    """
    zp1 = 1 + z
    return 0.015 * zp1**2.7 / (1 + (zp1 / 2.9)**5.6)


def cosmic_mstel(z, hubble_par, R=0.27):
    """ The cosmic stellar mass at redshift z in Msun / cubic Mpc

    (see figure 11 in Madau and Dickinson 2014).
    """
    def integrand(z):
        return cosmic_sfr(z) / (hubble_par(z).to(u.yr**-1).value * (z+1))

    result = quad(integrand, z, np.inf)[0]
    return (1 - R) * result
