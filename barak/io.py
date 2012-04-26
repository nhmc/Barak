""" Functions to read and write text, fits and pickle files.
"""
from itertools import izip
import cPickle as pickle
import os, gzip
import numpy as np
from utilities import adict

def readtxt(fh, sep=None, usecols=None, comment='#', skip=0,
            arrays=True, names=None, readnames=False, converters=None,
            mintype=int):
    """ Reads columns from a text file into arrays, converting to int,
    float or str where appropriate.

    By default the column separator is whitespace. `rows` can be
    either an input filename or an iterable (e.g. a file object, list
    or iterator).

    Parameters
    ----------
    rows : filename or iterable object
        Input data.
    sep : str (default `None`)
        A string used to separate items on a row (also known as a
        delimiter). Default is None, which means whitespace.
    usecols : int or tuple of ints, optional
        Indices of columns to be read. By default all columns are read.
    comment : str (default `#`)
        Character marking the start of a comment. 
    skip : int (default `0`)
        Number of rows to skip (not counting commented or blank lines)
        before reading data.
    arrays : bool (`True`)
        If True, all columns are converted to Numpy arrays.  If False,
        columns are returned as lists.
    names : str or sequence of str (default `None`)
        If `names` is given and `arrays` is True, the data are placed
        in a Numpy record array with field names given by `names`. Can
        also be a single string of comma-separated values.
    readnames : bool (`False`)
        If `readnames` is True the first line of the file is read to
        find the field names. This overrides the `names` keyword.
    converters : dict (`None`)
        Functions to apply to each entry of a column. Each (key,value)
        pair gives the column index (key) and the function to be
        applied to each entry in that column (value).

    Returns either structured array or lists.

    Examples
    --------
    >>> list_of_all_cols = readtxt('filename')
    >>> ninthcol, fifthcol = readtxt('filename', sep=',', usecols=(8,4)])
    >>> firstcol = readtxt('filename', comment='%', usecols=[0])
    >>> recarray = readtxt('filename', sep=',', usecols=(1,3), names='x,y'])
    """
    if mintype == float:
        typedict = {float : lambda x: str(x).strip()}
    elif mintype == int:
        typedict = {int : float,
                    float : lambda x: str(x).strip()}
    else:
        raise ValueError('Unknown minimum type %s' % mintype)

    def convert(row, funcs):
        # convert each item in a row to int, float or str.
        for i,item in enumerate(row):
            while True:
                try:
                    row[i] = funcs[i](item)
                except ValueError:
                    # update the list of converters
                    try:
                        funcs[i] = typedict[funcs[i]]
                    except KeyError:
                        raise ValueError('Converter %s failed '
                                         'on %r' % (funcs[i], item))
                else:
                    break
        return row,funcs

    needclose = False
    if isinstance(fh, basestring):
        if fh.endswith('.gz'):
            import gzip
            fh = gzip.open(fh)
        else:
            fh = open(fh)
        needclose = True

    data = iter(fh)

    if comment is not None:
        len_comment = len(comment)

    if names and isinstance(names, basestring):
        names = [n.strip() for n in names.split(',')]

    skipped = 0
    out = []
    # main loop to read data
    for irow, row in enumerate(data):
        if comment is not None:
            row = row.split(comment)[0]
        row = row.lstrip()
        if not row:  continue
        if skipped < skip:
            skipped += 1
            continue
        row = row.split(sep)
        if readnames:
            names = [r.strip() for r in row]
            readnames = False
            continue
        if not out:
            # first row with data, so initialise converters
            funcs = [mintype] * len(row)
            if converters is not None:
                for i in converters:
                    funcs[i] = converters[i]
            if usecols is not None:
                funcs = [funcs[i] for i in usecols]
        if usecols is not None:
            try:
                row = [row[i] for i in usecols]
            except IndexError:
                raise IndexError('Columns indices: %s, but only %i entries in '
                                 'this row!' % (usecols, len(row)))
        try:
            row, funcs = convert(row, funcs)
        except IndexError:
            # Probably there are more items in this row than in
            # previous rows. This usually indicates a problem, so
            # raise an error.
            raise IndexError('Too many items on row %i: %s' % (irow+1, row))

        if names:
            assert len(row) == len(names), '%i, %i, %s ' % (
                len(names), irow+1, row)
        out.append(row)

    if needclose:
        fh.close()

    # rows to columns, truncating to number of words on shortest line.
    if arrays:
        if names is not None:
            out = np.rec.fromrecords(out, names=names)
        else:
            out = [np.array(c) for c in izip(*out)]
    else:
        out = [list(c) for c in izip(*out)]

    if len(out) == 1 and names is None:
        return out[0]
    else:
        return out


def writetxt(fh, cols, sep=' ', names=None, header=None, overwrite=False,
             fmt_float='s'):
    """ Write data to a column-aligned text file.

    Structured array data written using this function can be read
    again using:

    >>> readtxt(filename, readnames=True)

    Parameters
    ----------
    fh :  file object or str
        The file to be written to.
    cols : structured array or a list of columns
        Data to be written.
    sep : str (' ')
        A string used to separate items on each row.
    names : list, string, False or None (None)
        Column names. Can be a comma-separated string of names. If
        False, do not print any names. If None and `cols` is a
        structured array, column names are the array field names.
    header : str (None)
        A header written before the data and column names.
    overwrite : bool (False)
        If True, overwrite an existing file without prompting.
    """
    # Open file (checking whether it already exists)
    if isinstance(fh, basestring):
        if not overwrite:
            while os.path.lexists(fh):
                c = raw_input('File %s exists, overwrite? (y)/n: ' % fh)
                if c == '' or c.strip().lower()[0] != 'n':
                    break
                else:
                    fh = raw_input('Enter new filename: ')
        fh = open(fh, 'w')

    if isinstance(names, basestring):
        names = names.split(',')

    try:
        recnames = cols.dtype.names
    except AttributeError:
        pass
    else:
        if names not in (None, False):
            recnames = names
        cols = [cols[n] for n in recnames]
        if names is None:
            names = list(recnames)

    cols = [np.asanyarray(c) for c in cols]

    if names not in (None, False):
        if len(names) < len(cols):
            raise ValueError('Need one name for each column!')

    nrows = [len(c) for c in cols]
    if max(nrows) != min(nrows):
        raise ValueError('All columns must have the same length!')
    nrows = nrows[0]

    # Get the maximum field width for each column, so that the columns
    # will line up when printed. Also find the printing format for
    # each column.
    maxwidths = []
    formats = []
    for col in cols:
        dtype = col.dtype.str[1:]
        if dtype.startswith('S'):
            maxwidths.append(int(dtype[1:]))
            formats.append('s')
        elif dtype.startswith('i'):
            maxwidths.append(max([len('%i' % i) for i in col]))
            formats.append('i')
        elif dtype.startswith('f'):
            maxwidths.append(max([len(('%' + fmt_float) % i) for i in col]))
            formats.append(fmt_float)
        elif dtype.startswith('b'):
            maxwidths.append(1)
            formats.append('i')
        else:
            raise ValueError('Unknown column data-type %s' % dtype)

    if names not in (None, False):
        for i,name in enumerate(names):
            maxwidths[i] = max(len(name), maxwidths[i])

    fmt = sep.join(('%-'+str(m)+f) for m,f in zip(maxwidths[:-1], formats[:-1]))
    fmt += sep + '%' + formats[-1] + '\n'

    if names:
        fmtnames = sep.join(('%-' + str(m) + 's') for m in maxwidths[:-1])
        fmtnames += sep + '%s\n'

    # Write the header if it was given
    if header is not None:
        fh.write(header)

    if names:
        fh.write(fmtnames % tuple(names))
    for row in izip(*cols):
        fh.write(fmt % tuple(row))

    fh.close()
    return

def writetabfits(filename, rec, units=None, overwrite=True):
    """ Writes a list of numpy arrays or a structured array to a
    binary fits table. Works best with structured arrays.

    Parameters
    ----------
    filename : str
      Filename to write to.
    rec : Sequence of arrays or record array
      Data to write.
    units : list of str (default None)
      Sequence of strings giving the units for each column.
    """
    import pyfits
    
    fmts = dict(f4='E', f8='F', i2='I', i4='J', i8='K', b1='L')

    try:
        rec.dtype
    except AttributeError:
        rec = np.rec.fromarrays(rec)
    if rec.dtype.names is None:
        raise ValueError('Input must be a list of columns or a '
                         'structured array')
    if units is None:
        units = [None] * len(rec.dtype.descr)

    cols = []
    for unit, name in zip(units, rec.dtype.names):
        a = rec[name]
        dtype = a.dtype.str[1:]
        if dtype.startswith('S'):
            fmt = 'A' + dtype[1:]
        else:
            fmt = fmts[dtype]
        cols.append(pyfits.Column(name=name, format=fmt, array=a, unit=unit))

    tbhdu = pyfits.new_table(pyfits.ColDefs(cols))
    tbhdu.writeto(filename, clobber=overwrite)

def readtabfits(filename, ext=None):
    """ Read fits binary table data, such as that written by
    `writetabfits()`.
    """
    import pyfits
    if ext is not None:
        return pyfits.getdata(filename, ext=ext).view(np.recarray)
    else:
        return pyfits.getdata(filename).view(np.recarray)

def saveobj(filename, obj, overwrite=False):
    """ Save a python object to filename using pickle."""
    if os.path.lexists(filename) and not overwrite:
        raise IOError('%s exists' % filename)
    if filename.endswith('.gz'):
        fh = gzip.open(filename, 'wb')
    else:
        fh = open(filename, 'wb')
    pickle.dump(obj, fh, protocol=2)
    fh.close()
    
def loadobj(filename):
    """ Load a python object pickled with saveobj."""
    if filename.endswith('.gz'):
        fh = gzip.open(filename, 'rb')
    else:
        fh = open(filename, 'rb')
    obj = pickle.load(fh)
    fh.close()
    return obj

def parse_config(filename, defaults={}):
    """ Read options for a configuration file.

    Parameters
    ----------
    filename : str
      The configuration filename.
    defaults : dict
      A dictionary with default values for options.

    The options are returned as a dictionary that can also be indexed
    by attribute.

    Ignores blank lines, lines starting with '#', and anything on a
    line after a '#'.

    Sample format::

     # this is the file with the line list
     lines = lines.dat    
     x = 20
     save = True    # save the data

    Attempts to convert the values to int, float, boolean otherwise
    string.
    """
    cfg = adict()
    
    cfg.update(defaults)

    fh = open(filename)
    for row in fh:
        if not row.strip() or row.lstrip().startswith('#'):
            continue
        option, value = [r.strip() for r in row.split('#')[0].split('=')]
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False

        cfg[option] = value

    fh.close()
    return cfg
