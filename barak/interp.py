""" Interpolation-related functions and classes.
""" 
import numpy as np
from utilities import between, indices_from_grid, meshgrid_nd

class CubicSpline(object):
    """ Class to generate a cubic spline through a set of points.

    After initialisation, an instance can be called with an array of
    values xp, and will return the cubic-spline interpolated values yp
    = f(xp).

    The spline can be reset to use a new first and last derivative
    while still using the same initial points by calling the set_d2()
    method.

    If you want to calculate a new spline using a different set of x
    and y values, you'll have to instantiate a new class.

    The spline generation is based on the NR algorithms (but not their
    code.)
    """
    def __init__(self, x, y, firstderiv=None, lastderiv=None, nochecks=False):
        """
        Parameters
        ----------
        x, y : arrays of floats, shape (N,)
          x and y = f(x).
        firstderiv : float (None)
          Derivative of f(x) at x[0]
        lastderiv : float (None)
          Derivative of f(x) at x[-1]
        nochecks : bool (False)
          If False, check the x array is sorted and unique. Set to True
          for increased speed.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not nochecks:
            # check all x values are unique
            if len(x) - len(np.unique(x)) > 0:
                raise Exception('non-unique x values were found!')
            cond = x.argsort()                    # sort arrays
            x = x[cond]
            y = y[cond]

        self.x = x
        self.y = y
        self.npts = len(x)
        self.set_d2(firstderiv, lastderiv)

    def __call__(self,xp):
        """ Given an array of x values, returns cubic-spline
        interpolated values yp = f(xp) using the derivatives
        calculated in set_d2().
        """
        x = self.x;  y = self.y;  npts = self.npts;  d2 = self.d2

        # make xp into an array
        if not hasattr(xp,'__len__'):  xp = (xp,)
        xp = np.asarray(xp)

        # for each xp value, find the closest x value above and below
        i2 = np.searchsorted(x,xp)

        # account for xp values outside x range
        i2 = np.where(i2 == npts, npts-1, i2)
        i2 = np.where(i2 == 0, 1, i2)
        i1 = i2 - 1

        h = x[i2] - x[i1]
        a = (x[i2] - xp) / h
        b = (xp - x[i1]) / h
        temp = (a**3 - a)*d2[i1] +  (b**3 - b)*d2[i2]
        yp = a * y[i1] + b * y[i2] + temp * h**2 / 6.

        return yp

    def _tridiag(self,temp,d2):
        x, y, npts = self.x, self.y, self.npts
        for i in range(1, npts-1):
            ratio = (x[i]-x[i-1]) / (x[i+1]-x[i-1])
            denom = ratio * d2[i-1] + 2.       # 2 if x vals equally spaced
            d2[i] = (ratio - 1.) / denom       # -0.5 if x vals equally spaced
            temp[i] = (y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
            temp[i] = (6.*temp[i]/(x[i+1]-x[i-1]) - ratio * temp[i-1]) / denom
        return temp

    def set_d2(self, firstderiv=None, lastderiv=None, verbose=False):
        """ Calculates the second derivative of a cubic spline
        function y = f(x) for each value in array x. This is called by
        __init__() when a new class instance is created.

        Parameters
        ----------
        firstderiv : float, (None)
          1st derivative of f(x) at x[0].  If None, then 2nd
          derivative is set to 0 ('natural').
        lastderiv : float (None)
          1st derivative of f(x) at x[-1].  If None, then 2nd
          derivative is set to 0 ('natural').
        """
        if verbose:  print 'first deriv,last deriv',firstderiv,lastderiv
        x, y, npts = self.x, self.y, self.npts
        d2 = np.empty(npts)
        temp = np.empty(npts-1)

        if firstderiv is None:
            if verbose:  print "Lower boundary condition set to 'natural'"
            d2[0] = 0.
            temp[0] = 0.
        else:
            d2[0] = -0.5
            temp[0] = 3./(x[1]-x[0]) * ((y[1]-y[0])/(x[1]-x[0]) - firstderiv)

        temp = self._tridiag(temp,d2)

        if lastderiv is None:
            if verbose:  print "Upper boundary condition set to 'natural'"
            qn = 0.
            un = 0.
        else:
            qn = 0.5
            un = 3./(x[-1]-x[-2]) * (lastderiv - (y[-1]-y[-2])/(x[-1]-x[-2]))

        d2[-1] = (un - qn*temp[-1]) / (qn*d2[-2] + 1.)
        for i in reversed(range(npts-1)):
            d2[i] = d2[i] * d2[i+1] + temp[i]

        self.d2 = d2

class AkimaSpline(object):
    """ A class used to generate an Akima Spline through a set of
    points.

    It must be instantiated with a set of `xvals` and `yvals` knot values,
    and then can be called with a new set of x values `x`. This is
    used by `interp_Akima`, see its documentation for more
    information.

    References
    ----------
    "A new method of interpolation and smooth curve fitting based
    on local procedures." Hiroshi Akima, J. ACM, October 1970, 17(4),
    589-602.

    Notes
    -----
    This is adapted from a function written by `Christoph Gohlke
    <http://www.lfd.uci.edu/~gohlke/>`_ under a BSD license:

    Copyright (c) 2007-2012, Christoph Gohlke
    Copyright (c) 2007-2012, The Regents of the University of California
    Produced at the Laboratory for Fluorescence Dynamics
    All rights reserved.
    """
    def __init__(self, xvals, yvals):
        """
        Parameters
        ----------
        xvals, yvals : array_like, shape (N,)
          Reference values. xvals cannot contain duplicates.
        """

        x = np.asarray(xvals, dtype=np.float64)
        y = np.asarray(yvals, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("x array must be one dimensional")
     
        n = len(x)
        if n < 3:
            raise ValueError("Array too small")
        if n != len(y):
            raise ValueError("Size of x-array must match data shape")
     
        dx = np.diff(x)
        if (dx <= 0.0).any():
            isort = np.argsort(x)
            x = x[isort]
            y = y[isort]
            dx = np.diff(x)
            if (dx == 0.).any():
                raise ValueError("x array has duplicate values")

        m = np.diff(y) / dx
        mm = 2. * m[0] - m[1]
        mmm = 2. * mm - m[0]
        mp = 2. * m[n - 2] - m[n - 3]
        mpp = 2. * mp - m[n - 2]
     
        m1 = np.concatenate(([mmm], [mm], m, [mp], [mpp]))
     
        dm = np.abs(np.diff(m1))
        f1 = dm[2:n + 2]
        f2 = dm[0:n]
        f12 = f1 + f2
     
        ids = np.nonzero(f12 > 1e-9 * f12.max())[0]
        b = m1[1:n + 1]
     
        b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
        c = (3. * m - 2. * b[0:n - 1] - b[1:n]) / dx
        d = (b[0:n - 1] + b[1:n] - 2. * m) / dx ** 2

        self.xvals, self.yvals, self.b, self.c, self.d = x, y, b, c, d

    def __call__(self, x):
        """
        Parameters
        ----------
        x : array_like, shape (M,)
          Values at which to interpolate.

        Returns
        -------
        vals : ndarray, shape (M,)
           Interpolated values.
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError("Array must be one dimensional")

        c0 = x < self.xvals[0]
        c2 = x > self.xvals[-1]
        c1 = ~(c0 | c2)
        x1 = x[c1]
        out = np.empty_like(x)
        bins = np.digitize(x1, self.xvals)
        bins = np.minimum(bins, len(self.xvals) - 1) - 1
        b = bins[0:len(x1)]
        wj = x1 - self.xvals[b]
        out[c1] = ((wj * self.d[b] + self.c[b])*wj + self.b[b])*wj + \
                  self.yvals[b]

        # use linear extrapolation for points outside self.xvals
        if c0.any():
            y = out[c1]
            slope = (y[1] - y[0]) / (x1[1] - x1[0])
            intercept = y[0] - slope * x1[0]
            out[c0] = x[c0] *slope + intercept
        if c2.any():
            y = out[c1]
            slope = (y[-2] - y[-1]) / (x1[-2] - x1[-1])
            intercept = y[-1] - slope * x1[-1]
            out[c2] = x[c2] *slope + intercept

        return out


def fit_spline(x, y, bins=4, estimator=np.median):
    """ Find a smooth function that approximates `x`, `y`.

    `bins` is the number of bins into which the sample is split. Returns
    a function f(x) that approximates y from min(x) to max(x).

    Notes
    -----
    The sample is split into bins number of sub-samples with evenly
    spaced x values. The median x and y value within each subsample is
    measured, and a cubic spline is drawn through these subsample
    median points.

    `x` must be sorted lowest -> highest, but need not be evenly spaced.
    """
    x,y = map(np.asarray, (x,y))
    good = ~np.isnan(x)
    good &= ~np.isnan(y)
    x = x[good]
    y = y[good]
    binedges = np.linspace(x.min(), x.max(), bins+1)
    medvals = []
    cbins = []
    for x0,x1 in zip(binedges[:-1], binedges[1:]):
        cond = between(x, x0, x1)
        if not cond.any():
            continue
        cbins.append(estimator(x[cond]))
        medvals.append(estimator(y[cond]))

    if len(cbins) < 3:
        raise RuntimeError('Too few bins')
    return AkimaSpline(cbins, medvals)

def interp_Akima(x_new, x, y):
    """Return interpolated data using Akima's method.

    Akima's interpolation method uses a continuously differentiable
    sub-spline built from piecewise cubic polynomials. The resultant
    curve passes through the given data points and will appear smooth
    and natural.

    Parameters
    ----------
    x_new : array_like, shape (M,)
        Values at which to interpolate.
    x, y : array_like, shape (N,)
        Reference values. x cannot contain duplicates.

    Returns
    -------
    vals : ndarray, shape (M,)
       Interpolated values.

    References
    ----------
    "A new method of interpolation and smooth curve fitting based
    on local procedures." Hiroshi Akima, J. ACM, October 1970, 17(4),
    589-602.

    Examples
    --------
    Plot interpolated Gaussian noise:

    >>> x = np.sort(np.random.random(10) * 100)
    >>> y = np.random.normal(0.0, 0.1, size=len(x))
    >>> x2 = np.arange(x[0], x[-1], 0.02)
    >>> y2 = interp_Akima(x2, x, y)
    >>> y3 = interp_spline(x2, x, y)
    >>> from matplotlib import pyplot as plt
    >>> plt.plot(x2, y2, 'b-', label='Akima')
    >>> plt.plot(x2, y3, 'r-', label='Cubic')
    >>> plt.plot(x, y, 'co')
    >>> plt.legend()
    >>> plt.show()
    """
    interpolator = AkimaSpline(x, y)
    return interpolator(x_new)

def interp_spline(x, xvals, yvals, nochecks=False):
    """ Like `numpy.interp`, but using spline instead of linear
    interpolation.

    This is a convenience function wrapped around CubicSpline.
    """
    spl = CubicSpline(xvals, yvals, nochecks=nochecks)
    return spl(x)

def splice(co0, co1, i, j, forced=None):
    """ Join two overlapping curves smoothly using a cubic spline.

    Parameters
    ----------
    co0, co1 : arrays of shape (N,)
      The two curves to be joined. They must have the same length and
      overlap completely.
    i, j : int
      Roughly speaking, co0 values will be retained below i, and co1 values
      will be retained above j.
    forced : int, optional
      The number of pixels and continuum values between i and j that
      continuum will be forced to pass through.

    Returns the new continuum array of shape (N,).
    """
    # go npix to either side of the joining point, and measure slopes
    newco = np.empty_like(co0)

    # derivatives
    d1 = co0[i] - co0[i-1]
    d2 = co1[j] - co1[j-1]

    if forced is not None:
        indices = [i] + list(zip(*forced)[0]) + [j]
        covals = [co0[i]] + list(zip(*forced)[1])+ [co1[j]]
    else:
        indices = [i,j]
        covals = [co0[i],co1[j]]

    indices = np.array(indices, dtype=float) 
    spline = CubicSpline(indices, covals,firstderiv=d1, lastderiv=d2)
    newco[:i] = co0[:i].copy()
    newco[i:j] = spline(range(i,j))
    newco[j:] = co1[j:].copy()

    return newco


def _interp3d(ix, iy, iz, a):
    """ Trilinear interpolation.

    Parameters
    ----------
    ix, iy, iz : arrays of floats, shape (M, )
      Coordinates of the points at which to interpolate. Values should
      be floats from 0 - I-1, 0 - J-1 and 0 - K-1.
    a: array of floats, shape (I, J, K)

    Returns
    -------
    output : array of floats, shape (M,)

    Notes
    -----
    This function isn't intended to be used directly. The public
    trilinear interpolation function is `trilinear_interp`.
    """
    assert len(ix) == len(iy) == len(iz)

    output = np.empty(len(ix))

    x0 = ix.astype(int)
    y0 = iy.astype(int)
    z0 = iz.astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x1[x1 == a.shape[0]] = a.shape[0] - 1
    y1[y1 == a.shape[1]] = a.shape[1] - 1
    z1[z1 == a.shape[2]] = a.shape[2] - 1

    x = ix - x0
    y = iy - y0
    z = iz - z0
    output = (a[x0,y0,z0] * (1-x) * (1-y) * (1-z) +
              a[x1,y0,z0] * x * (1-y) * (1-z) +
              a[x0,y1,z0] * (1-x) * y * (1-z) +
              a[x0,y0,z1] * (1-x) * (1-y) * z +
              a[x1,y0,z1] * x * (1-y) * z +
              a[x0,y1,z1] * (1-x) * y *z +
              a[x1,y1,z0] * x * y * (1-z) +
              a[x1,y1,z1] * x * y * z)

    return output

def trilinear_interp(x, y, z, xref, yref, zref, vals):
    """ Trilinear interpolation.

    Parameters
    ----------
    x, y, z : arrays of floats, shapes (M,), (N,), (O,)
      Coordinate grid at which to interpolate `vals`.
    xref, yref, zref : array of floats, shapes (I,), (J,), (K,)
      Reference coordinate grid. The grid must be equally spaced along
      each direction, but the spacing can be different between
      directions.
    vals : array of floats, shape (I, J, K)
      Reference values at the reference grid positions.

    Returns
    -------
    output : array of floats, shape (M, N, O)
    """
    assert (len(xref), len(yref), len(zref)) == vals.shape
    
    ix = indices_from_grid(x, xref)
    iy = indices_from_grid(y, yref)
    iz = indices_from_grid(z, zref)
    iX, iY, iZ = (a.ravel() for a in meshgrid_nd(ix, iy, iz))
    out = _interp3d(iX, iY, iZ, vals)

    # Note the index order and transpose!
    return out.reshape(len(z), len(y), len(x)).T

