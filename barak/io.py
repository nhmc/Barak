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
    """ This is deprecated. Use `writetable()` with file type '.tbl'
    instead.

    Write data to a column-aligned text file.

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
    """ This is deprecated. Use `writetable()` with file type '.fits'
    instead.

    Writes a list of numpy arrays or a structured array to a
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

    Consider using `atpy.Table(filename)` instead.
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
    filename : str or file object
      The configuration filename or a file object.
    defaults : dict
      A dictionary with default values for options.

    Returns
    -------
    d : dictionary
      The options are returned as a dictionary that can also be
      indexed by attribute.

    Notes
    -----
    Ignores blank lines, lines starting with '#', and anything on a
    line after a '#'. The parser attempts to convert the values to
    int, float or boolean, otherwise they are left as strings.

    Sample format::

     # this is the file with the line list
     lines = lines.dat
     x = 20
     save = True    # save the data
    """
    cfg = adict()

    cfg.update(defaults)

    if isinstance(filename, basestring):
        fh = open(filename)
    else:
        fh = filename

    for row in fh:
        if not row.strip() or row.lstrip().startswith('#'):
            continue
        option, value = [r.strip() for r in row.split('#')[0].split('=', 1)]
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
                elif value == 'None':
                    value = None

        cfg[option] = value

    fh.close()
    return cfg

def readsex(filename, catnum=None):
    """ Read a sextractor catalogue into a Numpy record array.

    Parameters
    ----------
    filename : str
      Sextractor output catalogue name
    catnum : int, optional
      If the Sextractor file is in LDAC_FITS format and contains more
      than one catalogue, this option specifies the catalogue number.

    Returns
    -------
    s : numpy record array
      Record array with field names the same as those in the
      sextractor catalogue.
    """
    fh = open(filename)
    # get the header
    row = fh.next()
    while not row.strip():
        row = fh.next()
    if row[8] == '=':
        fh.close()
        # assume a fits file
        try:
            import pyfits
        except ImportError:
            raise ValueError("Install Pyfits to read fits files")
        fh = pyfits.open(filename)
        if len(fh) > 3 and catnum is None:
            raise ValueError("specify catalogue number")
        elif catnum is not None:
            return pyfits.getdata(filename, catnum*2).view(np.recarray)
        else:
            return pyfits.getdata(filename, 2).view(np.recarray)
    hd = []
    while row.startswith('#'):
        if row[1:].strip():
            hd.append(row)
        row = fh.next()
    fh.close()
    # get column numbers and names
    number, names = zip(*[row.split() for row in hd])[1:3]
    indcol = [int(c)-1 for c in number]
    if len(names) - len(set(names)):
        dup = [n for n in set(names) if names.count(n) > 1]
        raise ValueError('fields with same names: %s' % dup)
    # read in the data
    return readtxt(filename, names=names, usecols=indcol)

def sex_to_DS9reg(filename, s, colour='green', tag='all', withtext=False):
    """Write a DS9 region file from SExtractor output.

    Parameters
    ----------
    filename : str
      Region file name.
    s : array
      The output of `readsex`.
    colour : str ('green')
      Region colour. One of {cyan blue magenta red green yellow white
      black}
    tag : str ('all')
      DS9 tag for all the regions
    with_text : bool (False)
      If True, then mark each region with either its magnitude (if
      given), otherwise its index in the input array `s`.
    """

    names = set(s.dtype.names)
    regions = ['global font="helvetica 10 normal" select=1 highlite=1 '
               'edit=0 move=1 delete=1 include=1 fixed=0 source']
    regions.append('image')
    fields = ['X_IMAGE', 'Y_IMAGE']
    if not ('X_IMAGE' in names and 'Y_IMAGE' in names):
        fields = ['XWIN_IMAGE', 'YWIN_IMAGE']
        if not ('XWIN_IMAGE' in names and 'YWIN_IMAGE' in names):
            raise ValueError('require either X_IMAGE and Y_IMAGE '
                             'or XWIN_IMAGE and YWIN_IMAGE')

    fmt = 'ellipse(%s %s %s %s %s) # text={%s} color=%s tag={%s}'
    ellipse_vals = ['A_IMAGE','B_IMAGE','THETA_IMAGE']
    ellipsewin_vals = ['AWIN_IMAGE','BWIN_IMAGE','THETAWIN_IMAGE']
    if all((n in names) for n in ellipse_vals):
        fields = list(fields) +  ellipse_vals
    elif all((n in names) for n in ellipsewin_vals):
        fields = list(fields) +  ellipsewin_vals
    else:
        # we don't have any ellipticity info, just write points.
        fmt = 'point(%s %s) # point=circle text={%s} color=%s tag={%s}'

    for i,rec in enumerate(s):
        vals = [rec[f] for f in fields]
        if withtext:
            if 'MAG_AUTO' in names:
                text = '%i %.2f' % (i+1, rec['MAG_AUTO'])
            else:
                text = i+1
        else:
            text = ''
        vals.extend([text, colour, tag])
        regions.append(fmt % tuple(vals))

    fh = open(filename,'w')
    fh.write('\n'.join(regions))
    fh.close()

def write_DS9reg(x, y, filename=None, coord='IMAGE', ptype='x', size=20,
                 c='green', tag='all', width=1, text=None):
    """Write a region file for ds9 for a  list of coordinates.

    Parameters
    ----------
    x, y : arrays of floats, shape (N,)
      The coordinates. These may be image or WCS.
    filename : str, optional
      A filename to write to.
    coord : str  ('IMAGE')
      The coordinate type. For example IMAGE (pixel coordinates) or
      J2000.
    ptype : str ('x')
      DS9 point type. One of {circle box diamond cross x arrow
      boxcircle}
    size : int (20)
      DS9 point size.
    c : str ('green')
      point colour: one of {cyan blue magenta red green yellow white
      black}.
    tag : str ('all')
      DS9 tag.
    width : int (1)
    """
    regions = ['global font="helvetica 10 normal" select=1 highlite=1 '
               'edit=0 move=1 delete=1 include=1 fixed=0 source\n']
    regions.append(coord + '\n')

    def iscontainer(s):
        try:
            it = iter(s)
        except TypeError:
            return False
        else:
            if isinstance(s, basestring) and len(s) != len(x):
                return False
        return True

    if not iscontainer(ptype):
        ptype = [ptype] * len(x)
    if not iscontainer(size):
        size = [size] * len(x)
    if not iscontainer(width):
        width = [width] * len(x)
    if not iscontainer(text):
        text = range(len(x))
    if not iscontainer(c):
        c = [c] * len(x)
    if not iscontainer(tag):
        tag = [tag] * len(x)

    fmt = ('point(%12.8f,%12.8f) # \
point=%s %s width=%s text={%s} color=%s tag={%s}\n')
    for i in xrange(len(x)):
        vals = (x[i], y[i], ptype[i], size[i], width[i], text[i],
                c[i], tag[i])
        regions.append(fmt % vals)

    if filename is not None:
        fh = open(filename,'w')
        fh.writelines(regions)
        fh.close()
    return regions

def writetable(filename, cols, units=None, names=None, header=None,
               keywords=None, overwrite=False):
    """ Write a series of data columns to a file.

    Data written using this function can be read again using:

    >>> atpy.Table(filename)

    Parameters
    ----------
    filename :  str
        The output filename. Its suffix determines the file type. For
        example '.tbl', '.fits' or '.fits.gz'.
    cols : structured array, atpy Table instance or a list of columns
        Data to be written.
    units : list
        Units of each column.
    names : list or string  (None)
        Column names. Can be a comma-separated string of names. If
        None and `cols` is a structured array, column names are the
        array field names.
    header : str (None)
        A header written before the data.
    keywords : dict (None)
        A dictionary of key-value pairs to write to the header.
    overwrite : bool (False)
        If True, overwrite an existing file without prompting.
    """
    import atpy

    if isinstance(cols, atpy.Table):
        t = cols
        old_formats = [t.columns[k].format for k in t.keys()]
    else:
        try:
            recnames = cols.dtype.names
        except AttributeError:
            assert np.allclose(len(cols[0]), [len(col) for col in cols[1:]])
        else:
            if names is not None:
                recnames = names
            else:
                names = list(recnames)
            cols = [cols[n] for n in recnames]

        if names is None:
            names = ['col%i' % (i+1) for i in range(len(cols))]
        elif isinstance(names, basestring):
            names = names.split(',')

        if units is None:
            units = [''] * len(names)

        t = atpy.Table()
        for i in xrange(len(cols)):
            t.add_column(names[i], cols[i], unit=units[i])

        if header is not None:
            for comment in header.split('\n'):
                t.add_comment(comment)

        if keywords is not None:
            for key,value in keywords.iteritems():
                t.add_keyword(key, value)

    if filename.endswith('.tbl') or filename.endswith('.tbl.gz'):
        # use str for int and floats to remove whitespace and make
        # easily-readable float values in IPAC tables - be warned this
        # may change the printed float values by about one part in
        # 1e12.
        for name in t.keys():
            if t.columns[name].format.endswith('s'):
                continue
            width = 0
            for item in t.data[name]:
                width = max(width, len(str(item)))
            t.columns[name].format = str(width) + 's'

    t.write(filename, overwrite=overwrite)

    if isinstance(cols, atpy.Table):
        # return column formats to their original values
        for fmt in old_formats:
            t.columns[name].format = fmt
