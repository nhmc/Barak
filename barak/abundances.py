""" Abundances and condensation temperatures.

Contains:

`Asolar`: an ordered dictionary of abundances from Lodders 2003, ApJ,
591, 1220. (http://adsabs.harvard.edu/abs/2003ApJ...591.1220L)

This gives a value A for each element, where A is defined for an
element el as::

  A(el) = log10 n(el)/n(H) - 12

n(el) is the number density of atoms of that element, and n(H) is the
number density of hydrogen.

and

`cond_temp`: an array of condensation temperatures (the temperature at
which an element in a gaseous state attaches to dust grains) for each
element from the same reference. In this array, `tc` is the
condensation temperature in K when condensation begins and `tc50` is
the 50% condensation temperature in K, when 50% of the element is left
in a gaseous state.
"""
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
