#!/usr/bin/env python
""" Crawl through all the modules and find the functions we want to
document. Then create an index.rst file for sphinx to process
autosummaries for.
""" 


import sys, os, inspect
from glob import glob

with open('../README.rst', 'r') as fh:
    readme = fh.read()

header = """\
Barak's Documentation
=====================
"""

header += readme




header += """\

.. raw:: html

 <a href="https://github.com/nhmc/Barak"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png" alt="Fork me on GitHub"></a>

.. toctree::
   :maxdepth: 2

   index.rst

"""

section_head = """\
%s
%s

%s

.. currentmodule:: %s

.. autosummary::
   :toctree: generated/

"""

footer = """\

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
"""

def is_mod_function(mod, func):
    return inspect.isfunction(func) and inspect.getmodule(func) == mod

def is_mod_class(mod, cla):
    return inspect.isclass(cla) and inspect.getmodule(cla) == mod

def list_functions(mod):
    """ List public functions for a module. """ 
    return [func.__name__ for func in mod.__dict__.itervalues()
            if is_mod_function(mod, func) and
            not func.__name__.startswith('_')]

def list_classes(mod):
    return [cla.__name__ for cla in mod.__dict__.itervalues()
            if is_mod_class(mod, cla)]

def list_methods(cla):
    """ List public methods for a class. """ 
    return [name for name,_ in inspect.getmembers(
        cla, predicate=inspect.ismethod) if not name.startswith('_')]

def parse_module(modname, prefix='barak.'):

    fullmodname = prefix + modname
    almostfullmodname = fullmodname[len('barak.'):]
    print fullmodname, modname
    if prefix == 'barak.':        
        mod = getattr(__import__(fullmodname), modname)
    else:
        mod = __import__(fullmodname, fromlist='barak')
    underline = '-' * len(almostfullmodname)
    moddoc = (mod.__doc__.strip() if mod.__doc__ is not None else '')
    s = section_head % (almostfullmodname, underline, moddoc, fullmodname)
    functions = sorted(list_functions(mod))
    classes = sorted(list_classes(mod))
    print '  functions', functions
    print '  classes', classes
    c = []
    for cla in classes:
        print '  class', cla
        try:
            c0 = getattr(mod, cla)
        except AttributeError:
            continue
        c.append(cla)
        methods = list_methods(c0)
        temp = [cla + '.' + n for n in sorted(set(methods))]
        c.extend(temp)

    for obj in c + functions:
        
        s += '   ' + obj + '\n'
    s += '\n'

    return s

def process_scripts(pkgdir, filenames):
    descriptions = []
    nwidth = dwidth = 0
    for n in filenames:
        print n
        fh = open(pkgdir + '/' + n)
        s = fh.read()
        fh.close()
        i = s.index('"""') + 3
        j = i + min(s[i:].index('"""'), s[i:].index('\n\n'))
        d = s[i:j].strip(' \n\\').replace('\n', ' ')
        descriptions.append(d)
        nwidth = max(len(n), nwidth)
        dwidth = max(len(d), dwidth)

    s = """\
Command line scripts
--------------------
"""
    s += '='*nwidth + ' ' + '='*dwidth + '\n'
    for i in range(len(filenames)):
        s += '%*s %-*s\n' % (nwidth, filenames[i], dwidth, descriptions[i])
    s += '='*nwidth + ' ' + '='*dwidth + '\n\n'

    return s

if 1:
    package_dir = '../barak'
    gen = os.walk(package_dir, topdown=1)
    # top level
    name, dirnames, filenames = gen.next()
    #print name, dirs, filenames

    filenames = [n for n in filenames if
                 n.endswith('.py') and n != '__init__.py']
    s = header

    for n in sorted(filenames):
        modname = n.replace('./', '')[:-3]
        s += parse_module(modname)

    # one more level down
    while True:
        try:
            name, temp, filenames = gen.next()
        except StopIteration:
            break
        
        # skip these directories, or if we're deeper than one level
        if name.startswith(package_dir + '/data') or \
           name.startswith(package_dir + '/tests') or \
           name.startswith(package_dir + '/sphinx') or \
           len(name[len(package_dir)+1:].split('/')) > 1:
            continue
            
        filenames = [n for n in filenames if
                     n.endswith('.py') and n != '__init__.py']
        prefix = 'barak.' + name[len(package_dir)+1:] + '.'

        s += parse_module(name[len(package_dir)+1:])

        print name, temp, filenames, prefix
        for n in sorted(filenames):
            modname = n.replace('./', '')[:-3]
            s += parse_module(modname, prefix=prefix)

    print 'scripts'
    filenames = sorted(glob('../scripts/*'))
    s += process_scripts('../scripts/', filenames)
    
    s += footer
    
    open('index.rst', 'w').write(s)
