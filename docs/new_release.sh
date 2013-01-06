# change the version number in 
#
#    docs/conf.py
#    setup.py
#    barak/__init__.py
#
# Update CHANGES

# make a new tag

git tag -a v0.3.0 -m'version 0.3.0'

# build tarball
python setup.py sdist

# test it installs correctly with
pip install ./Barak_vx.x.tar.gz
pip uninstall barak

# Then upload to PyPI (password is the usual)
python setup.py register sdist upload

# finally, check the PyPI site.  If everything went well, push the new tag.
git push --tags
