This package contains functions useful for scientific programming,
with a focus on astronomical research. The documentation details
everything that is available, but some example tasks that can be
handled are:

* Calculate absorption profiles for various ions observed in
  astrophysical environments.
* Fit a smooth continuum to a spectrum with many emission or
  absorption features.
* Find the expected broad-band magnitudes given a spectral energy
  distribution.

It requires `NumPy <http://numpy.scipy.org/>`_, `PyFits
<http://www.stsci.edu/institute/software_hardware/pyfits/Download>`_,
`ATpy <http://atpy.github.com/>`_ and `Astropy <http://astropy.org>`_
to install. `Matplotlib <http://matplotlib.sourceforge.net/>`_ and
`SciPy <http://www.scipy.org/>`_ are also required for full
functionality.

The best way to install is by using `pip`::

   pip install barak

(You may need to put a ``sudo`` in front of this). For this to work
you need to have `pip
<http://www.pip-installer.org/en/latest/index.html>`_ installed. This
method allows for easy uninstallation.

You can also simply download the tarball from the PyPI website, unpack
it and then do::

   python setup.py install

To run the tests you need `py.test <http://pytest.org/latest/>`_
installed.  Then run::

   py.test barak 

from the ``barak/`` directory created by the install commands above.

The latest stable package can be downloaded from PyPI: http://pypi.python.org/pypi/Barak.
The development version can be downloaded from `here <http://github.com/nhmc/Barak>`_.

Documentation is found here: http://nhmc.github.io/Barak/

Feel free to email me if you have any questions: 

neilcrighton .at. gmail .dot. com

