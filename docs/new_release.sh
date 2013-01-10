# change the version number in 
#
#    docs/conf.py
#    setup.py
#    barak/__init__.py
#
# Update CHANGES

# make a new tag

git tag -a v0.3.2 -m'version 0.3.2'

# build tarball
python setup.py sdist

# test it installs correctly with
sudo pip install dist/Barak-0.3.2.tar.gz
py.test
sudo pip uninstall barak

# Then upload to PyPI (password is the usual)
python setup.py register sdist upload

# finally, check the PyPI site.  If everything went well, push the new tag.
git push --tags
