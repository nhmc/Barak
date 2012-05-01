#!/usr/bin/env python
""" Crawl through all the modules and find the functions we want to
document. Then create an index.rst file for sphinx to process
autosummaries for.
""" 


import sys, os, inspect

header = """\
Welcome to Barak's documentation
================================

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
    c = []
    for cla in classes:
        c.append(cla)
        temp = [cla + '.' + n for n in sorted(
            set(list_methods(getattr(mod, cla))))]
        c.extend(temp)

    for obj in c + functions:
        s += '   ' + obj + '\n'
    s += '\n'

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
           name.startswith(package_dir + '/sphinx') or \
           name.startswith(package_dir + '/tests') or \
           name.startswith(package_dir + '/scripts') or \
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
        
    s += footer
    
    open('index.rst', 'w').write(s)
