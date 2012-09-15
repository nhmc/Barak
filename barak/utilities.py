""" Various general-use functions."""
from textwrap import wrap
import sys, os
import numpy as np
from math import sqrt

class Bunch(object):
    """Bunch class from the python cookbook with __str__ and __repr__
    methods.

    Examples
    --------
    >>> s = Bunch(a=1, b=2, c=['bar', 99])
    >>> s.a
    1
    >>> s.c
    ['bar', 99]
    >>> s
    Bunch(a, b, c)
    """
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    def __repr__(self):
        temp = ', '.join(sorted(str(attr) for attr in self.__dict__
                                if not str(attr).startswith('_')))
        return 'Bunch(%s)' % '\n      '.join(wrap(temp, width=69))

class adict(dict):
    """ A dictionary with attribute-style access. It maps attribute
    access to the real dictionary."""

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)

    # the following two methods allow pickling
    def __getstate__(self):
        """Prepare a state of pickling."""
        return self.__dict__.items()

    def __setstate__(self, items):
        """ Unpickle. """
        for key, val in items:
            self.__dict__[key] = val

    def __setitem__(self, key, value):
        return super(adict, self).__setitem__(key, value)

    def __getitem__(self, name):
        return super(adict, self).__getitem__(name)

    def __delitem__(self, name):
        return super(adict, self).__delitem__(name)

    def __setattr__(self, key, value):
        if hasattr(self, key):
            # make sure existing methods are not overwritten by new
            # keys.
            return super(adict, self).__setattr__(key, value)
        else:
            return super(adict, self).__setitem__(key, value)

    __getattr__ = __getitem__

    def copy(self):
        return adict(self)


def nan2num(a, replace=0):
    """ Replace `nan` or `inf` entries with the `replace` keyword
    value.

    If replace is "mean", use the mean of the array to replace
    values. If it's "interp", intepolate from the nearest values.
    """
    a = np.atleast_1d(a)
    b = np.array(a, copy=True)
    bad = np.isnan(b) | np.isinf(b)
    if replace == 'mean' and (~bad).sum() > 0:
        replace = b[~bad].mean().astype(b.dtype)
    elif replace == 'interp':
        x = np.arange(len(a))
        replace = np.interp(x[bad], x[~bad], b[~bad]).astype(b.dtype)
        
    b[bad] = replace
    if len(b) == 1:
        return b[0]
    return b

def indgroupby(a, name):
    """ Find the indices giving rows in `a` that have common values
    for the field `name`.

    Parameters
    ----------
    a : structured array
       Find the indices for rows in this array.
    name : str
       Field of `a`.

    Returns
    -------
    unique_vals, indices:
       unique sorted values of `a[name]`, and a list of indices into
       `a` giving the rows with each unique value.
    """
    b = a[name]
    isort = b.argsort()
    # find indices in the sorted array where we change to a new value
    # of name.
    temp = np.flatnonzero(b[isort][:-1] != b[isort][1:])
    jbreaks = np.concatenate([[0], temp + 1, [len(b)]])
    # make a list of indices into a for each unique value
    for j0,j1 in zip(jbreaks[:-1], jbreaks[1:]):
        ind = isort[j0:j1].view(np.ndarray)
        # yield the unique value, and the indices into `a` giving rows
        # with that value
        yield b[ind[0]], ind

def between(a, vmin, vmax):
    """ Return a boolean array True where vmin <= a < vmax.

    Notes
    -----
    Careful of floating point issues when dealing with equalities.
    """
    a = np.asarray(a)
    c = a < vmax
    c &= a >= vmin
    return c
    
def poisson_noise(nflux, nsigma, seed=None):
    """ Adds poisson noise to a normalised flux array.

    Parameters
    ----------
    nsigma : float
       One sigma error in the flux at the continuum level (where
       normalised flux=1).  
    nflux : array of floats, shape (N,)
       Normalised flux values (i.e. flux values divided by
       the continuum).
    seed : int, optional
      If given, this is used to seed the random number generator.

    Returns
    -------
    flux, error : arrays of floats, shape (N,)
      flux with noise added and one sigma error array.
    """

    if seed is not None:  np.random.seed(seed)

    nflux = np.asarray(nflux)
    nsigma = float(nsigma)
    if (nflux < 0).any():  raise Exception('Flux values must be >= 0!')
    lamb = nflux / (nsigma * nsigma)              # variance per pixel
    fout = np.random.poisson(lamb)
    fout = np.where(lamb > 0, fout/lamb * nflux, 0)
    sigout = np.where(lamb > 0, nflux/np.sqrt(lamb), 0)
    return fout, sigout

def addnoise(nflux, nsigma, minsig=None, seed=None):
    """ Add noise to a normalised flux array.

    Either gaussian, or a combination of poisson and gaussian noise is
    added to the flux array, depending on the keyword minsig.

    Parameters
    ----------
    nflux : array of floats, shape (N,)
      Array of normalised flux values.
    nsigma : float
      Total desired noise at the continuum (flux=1). Note the
      signal-to-noise ratio (SNR) = 1 / nsigma.
    minsig : float, optional
      By default minsig is `None`, which means gaussian noise with
      standard deviation `nsigma` is added to the flux. If minsig is
      given, a combination of poisson and gaussian noise is added to
      the flux to give an error of `sigma` at the continuum. In this
      case the gaussian noise component has st. dev. of `minsig`,
      which must be less than `nsigma`.
    seed : int, optional
      If seed is given, it is used to seed the random number generator.
      By default the seed is not reset.

    Returns
    -------
    flux, error : arrays of floats, shape (N,)
      Normalised flux with noise added, one sigma error array.
    """
    nsigma = abs(float(nsigma))
    if seed is not None:  np.random.seed(seed)

    if minsig is None:
        nflux = np.asarray(nflux)
        dev = nsigma * np.random.randn(len(nflux))
        er = np.empty_like(nflux)
        er.fill(nsigma)
        return nflux + dev, er
    else:
        minsig = abs(float(minsig))
        if minsig > nsigma:
            raise Exception('Noise at continuum must be bigger than minsig!')
        # gaussian variance
        var_g = minsig*minsig
        # normalised sigma of poisson noise at the continuum
        sig_p_cont = np.sqrt(nsigma*nsigma - var_g)

        flnew,sig_p = poisson_noise(nflux, sig_p_cont, seed=seed)
        # gaussian error
        flnew += minsig * np.random.randn(len(nflux))
        # total sigma
        er = np.sqrt(sig_p * sig_p + var_g)
        
        return flnew, er


def wmean(val, sig):
    """ Return the weighted mean and error. Uses inverse variances as
    weights.

    Parameters
    ----------
    val : array with shape (N,)
      Array of values.
    sig : array with shape (N,)
      One sigma errors (``sqrt(variance)``) of the array values.

    Returns
    -------
    wmean, wsigma : floats
       The weighted mean and error on the weighted mean.
    """
    val = np.asarray(val)
    sig = np.asarray(sig)

    # remove any values with bad errors
    condition = (sig > 0.) & ~np.isnan(val) & ~np.isnan(sig)
    if not condition.any():
        raise ValueError('No good values!')
    val = val[condition]
    sig = sig[condition]

    # normalisation
    inverse_variance = 1. / (sig*sig)
    norm = np.sum(inverse_variance)

    wmean = np.sum(inverse_variance*val) / norm
    wsigma = 1. / np.sqrt(norm)

    return wmean, wsigma


def indexnear(ar, val):
    """ Find the element in an array closest to a given value.

    The input array must be sorted lowest to highest.  Returns the
    index of the element with a value closest to the given value.

    Parameters
    ----------
    ar : array_like
      Input array. It must be sorted smallest to largest.
    val : float
      Find the element of `ar` that is closest to `val`.

    Returns
    -------
    index : int
       Index of the `ar` element with the closest value to `val`.

    Examples
    --------
    >>> wa = np.linspace(4000, 4500, 100)
    >>> i = indexnear(wa, 4302.5)
    >>> print i, wa[i]
    60 4303.03030303
    >>> i = indexnear(wa, 4600.0)
    >>> print i, wa[i]
    99 4500.0
    >>> i = indexnear(wa, 3000.0)
    >>> print i, wa[i]
    0 4000.0
    """

    ar = np.asarray(ar)
    i = ar.searchsorted(val)
    # following needed because searchsort rounds up
    if i == 0:
        return i
    # note if i == len(ar) then ar[i] is invalid, but won't get tested.
    elif i == len(ar) or val - ar[i-1] < ar[i] - val:
        return i-1
    else:
        return i


def calc_Mstar_b(z):
    """ Find M* at a given redshift.

    Find the Schechter parameter M* in the rest frame B band at
    redshift z, by interpolating over the Faber at al. 2007 DEEP2
    averaged values, and assuming M*_B = -20.0 at z=0 (rough average
    of the z=0.07-0.1 points in Faber 2007) .
    """
    assert z < 1.5
    zvals = 0.0, 0.3, 0.5, 0.7, 0.9, 1.1
    Mvals = -20.00, -21.07, -21.15, -21.51, -21.36, -21.54
    return np.interp(z, zvals, Mvals)


def combinations(items, n):
    """ A generator of combinations from `items`.

    This returns a generator for the number of ways you can take n
    items (order unimportant) from a list of items."""
    if n == 0:
        yield []
    else:
        for i in xrange(len(items)):
            for c in combinations(items[i+1:], n-1):
                yield [items[i]] + c

def permutations(items):
    """ Generator for permutations from `items`.

    These are a special case of `combinations`.
    """
    return combinations(items, len(items))

def find_edges_true_regions(condition):
    """ Finds the indices for the edges of contiguous regions where
    condition is True.

    Examples
    --------
    >>> a = np.array([3,0,1,4,6,7,8,6,3,2,0,3,4,5,6,4,2,0,2,5,0,3])
    >>> ileft, iright = find_edges_true_regions(a > 2)
    >>> zip(ileft, iright)
    [(0, 0), (3, 8), (11, 15), (19, 19), (21, 21)]

    """
    indices, = condition.nonzero()
    if not len(indices):
        return None, None
    iright, = (indices[1:] - indices[:-1] > 1).nonzero()
    ileft = iright + 1
    iright = np.concatenate( (iright, [len(indices)-1]) )
    ileft = np.concatenate( ([0], ileft) )
    return indices[ileft], indices[iright]

def stats(arr):
    """ Show the minimum, maximum median, mean, shape and size of an
    array.

    Also show the number of NaN entries (if any).
    """    
    arr = np.asarray(arr)
    shape = arr.shape
    arr = arr.ravel()
    size = len(arr)
    bad = np.isnan(arr)
    nbad = bad.sum()
    if nbad == size:
        return '#NaN %i of %i' % (nbad, size)
    elif nbad == 0:
        arr = np.sort(arr)
    else:
        arr = np.sort(arr[~bad])
    if len(arr) % 2 == 0:
        i = len(arr) // 2
        median = 0.5 * (arr[i-1] + arr[i])
    else:
        median = arr[len(arr) // 2]

    return 'min %.5g max %.5g median %.5g mean %.5g shape %s #NaN %i of %i' % (
        arr[0], arr[-1], median, arr.mean(), shape, nbad, size)

def Gaussian(x, x0, sigma, height):
    """ Gaussian."""
    return height * np.exp(-0.5 * ((x-x0)/sigma)**2)

def meshgrid_nd(*arrs):
    """ Like numpy's meshgrid, but works on more than two dimensions.
    """
    arrs = tuple(reversed(arrs))
    lens = map(len, arrs)
    dim = len(arrs)

    sz = 1
    for s in lens:
        sz*=s

    ans = []    
    for i, arr in enumerate(arrs):
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j) 
        ans.append(arr2)

    return tuple(ans[::-1])

def autocorr(x, maxlag=300):
    """ Find the autocorrelation of x.

    The mean is subtracted from x before correlating. Correlation
    values are calculated in offset steps from 0 up to maxlag.
    """
    dot = np.dot
    maxlag = min(maxlag, len(x)-1)
    x = np.asanyarray(x)
    x = x - x.mean()
    a = [1]
    for k in xrange(1, maxlag):
        v1 = dot(x[:-k], x[k:])
        v2 = dot(x[:-k], x[:-k])
        v3 = dot(x[k:], x[k:])
        a.append(v1 / sqrt(v2*v3))

    return a

def get_data_path():
    """ Return the path to the data directory for this package.
    """
    return os.path.abspath(__file__).rsplit('/', 1)[0] + '/data/'

def indices_from_grid(c, ref):
    """ Convert coordinates to indices defined by grid of reference
    values.

    Parameters
    ----------
    c : array of floats, shape (M,)
      Coordinates.
    ref : array of floats, shape (N,)
      Reference grid coordinates. They must be equally spaced.

    Returns
    -------
    ind : arrays of floats
      Coordinates mapped onto the indices of the reference grid.
    """
    ref = np.sort(ref)

    dref = ref[1:] - ref[:-1]
    dref0 = float(dref[0])
    assert np.allclose(dref0, dref[1:])

    c = np.sort(c)
    
    assert c[0] >= ref[0] and c[-1] <= ref[-1] 

    ind = (c - ref[0]) / dref0

    return ind
