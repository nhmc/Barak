# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
barak.cosmology contains routines for cosmological distance measures
and other quantities. References for most of the quantities calculated
in this package are given by Hogg (astro-ph/9905116).

The following objects are available in the barak.cosmology namespace:

  Cosmology, kpc_comoving_per_arcmin, kpc_proper_per_arcmin,
  arcsec_per_kpc_comoving, arcsec_per_kpc_proper, distmod,
  radec_to_xyz, get_default, set_default, WMAP5, WMAP7

The code for these lives in the barak.cosmology.core module.

Most of the functionality is enabled by the `Cosmology` object. To
create a new `Cosmology` object with arguments giving the hubble
parameter, omega matter and omega lambda (all at z=0)::

  >>> from barak.cosmology import Cosmology
  >>> cosmo = Cosmology(H0=70, Om=0.3, Ol=0.7)
  >>> cosmo
  Cosmology(H0=70, Om=0.3, Ol=0.7, Ok=0)

The methods of this object calculate commonly used quantities with
your cosmology. So the comoving distance in Mpc at redshift 4 is given
by:

  >>> cosmo.comoving_distance(4)
  7170.366414463296

The age of the universe at z = 0 in Gyr::

  >>> cosmo.age(0)
  13.46698402784007

See the `Cosmology` object docstring for all the methods and variables
available.  There are several standard cosmologies already defined:

  >>> from cosmology import WMAP7    # Parameters from the 7-year WMAP results
  >>> WMAP7.critical_density0        # Critical density at z=0 in g/cm^3
  9.31000313202047e-30

  >>> from cosmology import WMAP5    # From the 5-year WMAP results
  >>> WMAP5.H(3)                     # Hubble parameter at z=3 in km/s/Mpc
  301.54148311633674

There are also several convenience functions that calculate quantities
without needing to create a Cosmology object. These use a 'default'
cosmology if no cosmology instance is explicitly passed to them. The
default can be set with `cosmology.set_default()` and is accessed with
`cosmology.get_default()`. If you don't set the default explicitly,
then the first time it is accessed a warning message is printed and
it's set to the 7 year WMAP cosmology.

  >>> from astropy import cosmology
  >>> cosmology.kpc_proper_per_arcmin(3)
  472.91882815884907
  >>> cosmology.arcsec_per_kpc_proper(3)
  0.12687166682195736

"""
from core import \
     Cosmology, kpc_comoving_per_arcmin, kpc_proper_per_arcmin, \
     arcsec_per_kpc_comoving, arcsec_per_kpc_proper, distmod, radec_to_xyz, \
     get_default, set_default, WMAP5, WMAP7

