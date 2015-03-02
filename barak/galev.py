"""
Galaxy-evolution related utilities

"""
import astropy.units as u
import numpy as np
from scipy.integrate import quad

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
