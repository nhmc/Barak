""" Abundances and condensation temperatures. """
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
