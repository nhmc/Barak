""" Abundances and condensation temperatures.

This contains the following two datasets:

Asolar:
  An ordered dictionary of abundances from `Lodders 2003, ApJ, 591,
  1220 <http://adsabs.harvard.edu/abs/2003ApJ...591.1220L>`_. It
  contains a value A for each element `el`, where A is defined::

    A(el) = log10 n(el)/n(H) + 12

  `n(el)` is the number density of atoms of that element, and `n(H)`
  is the number density of hydrogen.

cond_temp:
  An array of condensation temperatures for each element from the same
  reference.  The condensation temperature is the temperature at which
  an element in a gaseous state attaches to dust grains.

  It contains the values `tc` and `tc50` for each element, where `tc`
  is the condensation temperature in K when condensation begins, and
  `tc50` is the temperature when 50% of the element is left in a
  gaseous state.

"""
import numpy as np
from utilities import get_data_path
from io import readtxt
from collections import OrderedDict

datapath = get_data_path()

Asolar = OrderedDict(
    (t.el, t.A) for t in
    readtxt(datapath + 'abundances/SolarAbundance.txt', readnames=1))

cond_temp = readtxt(datapath +
                    'abundances/CondensationTemperatures.txt',
                    readnames=1, sep='|')

def calc_abund(X, Y, logNX, logNY):
    """ Find the abundance relative to solar given two elements and
    their column densities.

    Parameters
    ----------
    X, Y : str
      Element identifiers (for example 'C', 'Si', 'Mg').
    logNX : array_like, shape (N,)
      log10 of element X column density in cm^-2.
    logNY : array_like, shape (N,)
      log10 of element Y column density in cm^-2.

    Returns
    -------
    abundance_ratio : ndarray, shape (N,)
       Abundance ratio relative to solar, [X/Y].

    Notes
    -----
    The abundance ratio is defined::

      [X/Y] = log10 (n_X / n_Y) - log10 (n_Xsun / n_Ysun)
    
    Where N_Xsun / N_Ysun is the ratio of the number density of
    species X to species Y for 'proto-solar' abundances (See Lodders
    2003, ApJ, 591, 1220). For example, if [X/Y] = 0, it has the same
    abundance as the proto-solar values. If [X/Y] = 1, then ten times
    larger, [X/Y] = -1, then ten times smaller.

    If Y is hydrogen, then this estimates the metallicity.
    """
    Z = np.log10(10**logNX / 10**logNY) - (Asolar[X] - Asolar[Y])
    return  Z
