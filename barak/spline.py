""" Spline-related functions.
""" 
import numpy as np
from utilities import between

class InterpCubicSpline:
    """Interpolate a cubic spline through a set of points.

    After initialisation, an instance can be called with an array of
    values xp, and will return the cubic-spline interpolated values yp
    = f(xp).

    The spline can be reset to use a new first and last derivative
    while still using the same initial points by calling the set_d2()
    method.

    If you want to calculate a new spline using a different set of x
    and y values, you'll have to instantiate a new class.

    The spline generation is based on the NR algorithms (but not their
    routines.)
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
        x = np.asarray(x)
        y = np.asarray(y)
        if 'i' in  (x.dtype.str[1], y.dtype.str[1]):
            raise TypeError('Input arrays must not be integer')
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

def interp_spline(x, xvals, yvals, nochecks=False):
    """ Like `numpy.interp`, but using spline instead of linear
    interpolation.

    This is a convenience function wrapped around InterpCubicSpline.
    """
    spl = InterpCubicSpline(xvals, yvals, nochecks=nochecks)
    return spl(x)

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
    return InterpCubicSpline(cbins, medvals, nochecks=1)


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
    spline = InterpCubicSpline(indices, covals,firstderiv=d1, lastderiv=d2)
    newco[:i] = co0[:i].copy()
    newco[i:j] = spline(range(i,j))
    newco[j:] = co1[j:].copy()

    return newco
