# change the version number in 
#
#    docs/conf.py
#    setup.py
#    barak/__init__.py
#
# Update CHANGES

# make a new tag

git tag -a v0.3.0 -m'version 0.3.0'
git push --tags

# build tarball
python setup.py sdist

# test it installs correctly with
pip install ./Barak_vx.x.tar.gz
pip uninstall barak

# Then upload to PyPI
python setup.py register sdist upload

