
from distutils.core import setup
from glob import glob
import os, sys
import warnings

# To create a distribution, use
#
#     python setup.py sdist
#
# Register on PyPI
#
#     python setup.py register
#
# Upload the distribution to PyPI
#
#     python setup.py register sdist upload

def get_data_names(root):
    """ Return list of all filenames (not directories) under root.
    """
    temp = []
    for dirpath, dirnames, filenames in os.walk(root):
        temp.extend((os.path.join(dirpath, d, '*') for d in dirnames))

    names = []
    for path in temp:
        if any(os.path.isfile(f) for f in glob(path)):
            names.append(path[6:])

    return names


if len(sys.argv[1:]) > 0 and sys.argv[1] in ('build', 'install'):
    try:
        import matplotlib
    except ImportError:
        warnings.warn("""
matplotlib doesn't look like it's been installed.  I'm continuing to
install barak, but be warned that some modules and scripts will not
work. To install matplotlib, follow the instructions at
http://matplotlib.org/users/installing.html.

""")
    

with open('README') as fh:
    readme = fh.read()

description = ("A set of astronomy-related routines for generating Voigt "
               "profiles from atomic data, reading and writing data, "
               "working with SEDs and passbands.")

package_data = {'barak' : get_data_names('barak/data')}

setup(
    name = 'Barak',
    version = '0.2.0',
    author = 'Neil Crighton',
    author_email = 'neilcrighton .at. gmail .dot. com',
    packages = ['barak'],
    package_dir = {'barak': 'barak'},
    package_data = package_data,
    include_package_data = True,
    scripts = glob('scripts/*'),
    license = 'LICENCE',
    url = 'http://pypi.python.org/pypi/Barak/',
    description = description,
    long_description = readme,
    requires = ["numpy", "pyfits", "atpy"],
    install_requires = ["numpy", "pyfits", "atpy"]
    )
