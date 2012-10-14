from distutils.core import setup
from glob import glob
import os

def get_data_names(root):

    temp = []
    for dirpath, dirnames, filenames in os.walk(root):
        temp.extend((os.path.join(dirpath, d, '*') for d in dirnames))

    names = []
    for path in temp:
        if any(os.path.isfile(f) for f in glob(path)):
            names.append(n[6:])

    return names


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
    packages = ['barak'],
    package_dir = {'barak': 'barak'},
    package_data = package_data,
    include_package_data = True,
    scripts = glob('scripts/*'),
    license = 'LICENCE',
    url = 'http://pypi.python.org/pypi/Barak/',
    description = description,
    long_description = readme,
    requires = ["numpy", "pyfits"]
    )
