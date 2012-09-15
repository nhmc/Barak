""" Plotting routines. """
from __future__ import division

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import PolyCollection, LineCollection
import matplotlib.transforms as mtransforms

A4LANDSCAPE = 11.7, 8.3
A4PORTRAIT = 8.3, 11.7

def default_marker_size(fmt):
    """ Find a default matplotlib marker size such that different marker types
    look roughly the same size.
    """
    temp = fmt.replace('.-', '')
    if '.' in temp:
        ms = 10
    elif 'D' in temp:
        ms = 7
    elif set(temp).intersection('<>^vd'):
        ms = 9
    else:
        ms = 8
    return ms

def axvfill(xvals, ax=None, color='k', alpha=0.1, edgecolor='none', **kwargs):
    """ Fill vertical regions defined by a sequence of (left, right)
    positions.

    Parameters
    ----------
    xvals: list
      Sequence of pairs specifying the left and right extent of each
      region. e.g. (3,4) or [(0,1), (3,4)]
    ax : matplotlib axes instance (default is the current axes)
      The axes to plot regions on.
    color : mpl colour (default 'g')
      Color of the regions.
    alpha : float (default 0.3)
      Opacity of the regions (1=opaque).

    Other keywords arguments are passed to PolyCollection.
    """
    if ax is None:
        ax = pl.gca()
    xvals = np.asanyarray(xvals)
    if xvals.ndim == 1:
        xvals = xvals[None, :]
    if xvals.shape[-1] != 2:
        raise ValueError('Invalid input')

    coords = [[(x0,0), (x0,1), (x1,1), (x1,0)] for x0,x1 in xvals]
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    kwargs.update(facecolor=color, edgecolor=edgecolor, transform=trans, alpha=alpha)

    p = PolyCollection(coords, **kwargs)
    ax.add_collection(p)
    ax.autoscale_view()
    return p

def axvlines(xvals, ymin=0, ymax=1, ax=None, ls='-', color='0.7', **kwargs):
    """ Plot a set of vertical lines at the given positions.
    """
    if ax is None:
        ax = pl.gca()

    coords = [[(x,ymin), (x,ymax)] for x in xvals]
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    kwargs.update(linestyle=ls, colors=color, transform=trans)

    l = LineCollection(coords, **kwargs)
    ax.add_collection(l)
    ax.autoscale_view()
    return l


def puttext(x,y,text,ax, xcoord='ax', ycoord='ax', **kwargs):
    """ Print text on an axis using axes coordinates."""
    if xcoord == 'data' and ycoord == 'ax':
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    elif xcoord == 'ax' and ycoord == 'data':
        trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    elif xcoord == 'ax' and ycoord == 'ax':
        trans = ax.transAxes
    else:
        raise ValueError("Bad keyword combination: %s, %s "%(xcoord,ycoord))
    return ax.text(x, y, str(text), transform=trans, **kwargs)


def distplot(vals, xvals=None, perc=(68, 95), showmean=False,
             showoutliers=True, color='forestgreen',  ax=None,
             logx=False, logy=False, negval=None, **kwargs):
    """
    Make a top-down histogram plot for an array of
    distributions. Shows the median, 68%, 95% ranges and outliers.

    Similar to a boxplot.

    Parameters
    ----------
    vals : sequence of arrays
        2-d array or a sequence of 1-d arrays.
    xvals : array of floats
        x positions.
    perc : array of floats  (68, 95)
        The percentile levels to use for area shading. Defaults show
        the 68% and 95% percentile levels; roughly 1 and 2
        sigma ranges for a Gaussian distribution.
    showmean : boolean  (False)
        Whether to show the means as a dashed black line.
    showoutliers : boolean (False)
        Whether to show outliers past the highest percentile range.
    color : mpl color ('forestgreen')
    ax : mpl Axes object
        Plot to this mpl Axes instance.
    logx, logy : bool (False)
        Whether to use a log x or y axis.
    negval : float (None)
        If using a log y axis, replace negative plotting values with
        this value (by default it chooses a suitable value based on
        the data values).
    """
    if any(not hasattr(a, '__iter__') for a in vals):
        raise ValueError('Input must be a 2-d array or sequence of arrays')

    assert len(perc) == 2
    perc = sorted(perc)
    temp = 0.5*(100 - perc[0])
    p1, p3 = temp, 100 - temp
    temp = 0.5*(100 - perc[1])
    p0, p4 = temp, 100 - temp
    percentiles = p0, p1, 50, p3, p4

    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)

    if xvals is None:
        xvals = np.arange(len(vals), dtype=float)


    # loop through columns, finding values to plot
    x = []
    levels = []
    outliers = []
    means = []
    for i in range(len(vals)):
        d = np.asanyarray(vals[i])
        # remove nans
        d = d[~np.isnan(d)]
        if len(d) == 0:
            # no data, skip this position
            continue
        # get percentile levels
        levels.append(scoreatpercentile(d, percentiles))
        if showmean:
            means.append(d.mean())
        # get outliers
        if showoutliers:
            outliers.append(d[(d < levels[-1][0]) | (levels[-1][4] < d)])
        x.append(xvals[i])

    levels = np.array(levels)
    if logx and logy:
        ax.loglog([],[])
    elif logx:
        ax.semilogx([],[])
    elif logy:
        ax.semilogy([],[])

    if logy:
        # replace negative values with a small number, negval
        if negval is None:
            # guess number, falling back on 1e-5
            temp = levels[:,0][levels[:,0] > 0]
            if len(temp) > 0:
                negval = np.min(temp)
            else:
                negval = 1e-5

        levels[~(levels > 0)] = negval
        for i in range(len(outliers)):
            outliers[i][outliers[i] < 0] = negval
            if showmean:
                if means[i] < 0:
                    means[i] = negval

    ax.fill_between(x,levels[:,0], levels[:,1], color=color, alpha=0.2, edgecolor='none')
    ax.fill_between(x,levels[:,3], levels[:,4], color=color, alpha=0.2, edgecolor='none')
    ax.fill_between(x,levels[:,1], levels[:,3], color=color, alpha=0.5, edgecolor='none')
    if showoutliers:
        x1 = np.concatenate([[x[i]]*len(out) for i,out in enumerate(outliers)])
        out1 = np.concatenate(outliers)
        ax.plot(x1, out1, '.', ms=1, color='0.3')
    if showmean:
        ax.plot(x, means, 'k--')
    ax.plot(x, levels[:,2], 'k-', **kwargs)
    ax.set_xlim(xvals[0],xvals[-1])
    try:
        ax.minorticks_on()
    except AttributeError:
        pass

    return ax

def errplot(x, y, yerrs, xerrs=None, fmt='.b', ax=None, ms=None, mew=0.5,
            ecolor=None, elw=None, zorder=None, nonposval=None, **kwargs):
    """ Plot a graph with errors.

    Parameters
    ----------
    x, y : arrays of shape (N,)
        Data.
    yerrs : array of shape (N,) or (N,2)
        Either an array with the same length as `y`, or a list of two
        such arrays, giving lower and upper limits to plot.
    xerrs : array, shape (N,) or (N,2), optional
        Optional x errors. The format is the same as for `yerrs`.
    fmt : str
        A matplotlib format string that is passed to `pylab.plot`.
    ms, mew : floats
        Plotting marker size and edge width.
    ecolor : matplotlib color (None)
        Color of the error bars. By default this will be the same color
        as the markers.
    elw: matplotlib line width (None)
        Error bar line width.
    nonposval : float (None)
        Replace any non-positive values of y with `nonposval`.
    """

    if ax is None:
        fig = pl.figure()
        ax = fig.add_subplot(111)

    yerrs = np.asarray(yerrs)
    if yerrs.ndim > 1:
        lo = yerrs[0]
        hi = yerrs[1]
    else:
        lo = y - yerrs
        hi = y + yerrs

    if nonposval is not None:
        y = np.where(y <= 0, nonposval, y)

    if ms is None:
        ms = default_marker_size(fmt)

    l, = ax.plot(x, y, fmt, ms=ms, mew=mew, **kwargs)
    # find the error colour
    if ecolor is None:
        ecolor = l.get_mfc()
        if ecolor == 'none':
            ecolor = l.get_mec()
    if nonposval is not None:
        lo[lo <= 0] = nonposval
        hi[hi <= 0] = nonposval

    if 'lw' in kwargs and elw is None:
        elw = kwargs['lw']
    col = ax.vlines(x, lo, hi, color=ecolor, lw=elw, label='__nolabel__')

    if xerrs is not None:
        xerrs = np.asarray(xerrs)
        if xerrs.ndim > 1:
            lo = xerrs[0]
            hi = xerrs[1]
        else:
            lo = x - xerrs
            hi = x + xerrs
        col2 = ax.hlines(y, lo, hi, color=ecolor, lw=elw, label='__nolabel__')

    if zorder is not None:
        col.set_zorder(zorder)
        l.set_zorder(zorder)
        if xerrs is not None:
            col2.set_zorder(zorder)

    if pl.isinteractive():
        pl.show()

    return ax

def dhist(xvals, yvals, xbins=20, ybins=20, ax=None, c='b', fmt='.', ms=1,
          label=None, loc='right,bottom', xhistmax=None, yhistmax=None,
          histlw=1, xtop=0.2, ytop=0.2, chist=None, **kwargs):
    """ Given two set of values, plot two histograms and the
    distribution.

    xvals,yvals are the two properties to plot.  xbins, ybins give the
    number of bins or the bin edges. c is the color.
    """

    if chist is None:
        chist = c
    if ax is None:
        pl.figure()
        ax = pl.gca()

    loc = [l.strip().lower() for l in loc.split(',')]

    if ms is None:
        ms = default_marker_size(fmt)

    ax.plot(xvals, yvals, fmt, color=c, ms=ms, label=label, **kwargs)
    x0,x1,y0,y1 = ax.axis()

    if np.__version__ < '1.5':
        x,xbins = np.histogram(xvals, bins=xbins, new=True)
        y,ybins = np.histogram(yvals, bins=ybins, new=True)
    else:
        x,xbins = np.histogram(xvals, bins=xbins)
        y,ybins = np.histogram(yvals, bins=ybins)

    b = np.repeat(xbins, 2)
    X = np.concatenate([[0], np.repeat(x,2), [0]])
    Xmax = xhistmax or X.max()
    X = xtop * X / Xmax
    if 'top' in loc:
        X = 1 - X
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(b, X, color=chist, transform=trans, lw=histlw)

    b = np.repeat(ybins, 2)
    Y = np.concatenate([[0], np.repeat(y,2), [0]])
    Ymax = yhistmax or Y.max()
    Y = ytop * Y / Ymax
    if 'right' in loc:
        Y = 1 - Y
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot(Y, b, color=chist, transform=trans, lw=histlw)

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
    if pl.isinteractive():
        pl.show()

    return ax, dict(x=x, y=y, xbinedges=xbins, ybinedges=ybins)

def histo(a, fmt='b', bins=10, ax=None, lw=2, log=False, **kwargs):
    """ Plot a histogram, without all the unnecessary stuff
    matplotlib's hist() function does."""

    if ax is None:
        pl.figure()
        ax = pl.gca()

    vals,bins = np.histogram(np.asarray(a).ravel(), bins=bins)
    if log:
        vals = np.where(vals > 0, np.log10(vals), vals)
    b = np.repeat(bins, 2)
    V = np.concatenate([[0], np.repeat(vals,2), [0]])
    ax.plot(b, V, fmt, lw=lw, **kwargs)
    if pl.isinteractive():
        pl.show()
    return vals,bins

def arrplot(a, x=None, y=None, ax=None, perc=(0, 100), colorbar=True,
            **kwargs):
    """ Plot a 2D array with coordinates.

    Label coordinates such that each coloured patch representing a
    value in `a` is centred on its x,y coordinate.

    Parameters
    ----------
    a : array, shape (N, M)
      Values at each coordinate.
    x : shape (N,)
      Coordinates, must be equally spaced.
    y : shape (M,)
      Coordinates, must be equally spaced.
    ax : axes
      Axes in which to plot.
    colorbar : bool (True)
      Whether to also plot a colorbar.
    """
    if x is None:
        x = np.arange(a.shape[0])
    if y is None:
        y = np.arange(a.shape[1])

    assert len(x) == a.shape[0]
    assert len(y) == a.shape[1]

    if ax is None:
        pl.figure()
        ax = pl.gca()
    
    assert np.allclose(x, np.sort(x))
    assert np.allclose(y, np.sort(y))

    dxvals = x[1:] - x[:-1]
    dx = dxvals[0]
    assert np.allclose(dx, dxvals[1:])
    x0, x1 = x[0] - 0.5*dx, x[-1] + 0.5*dx

    dyvals = y[1:] - y[:-1]
    dy = dyvals[0]
    assert np.allclose(dy, dyvals[1:])
    y0, y1 = y[0] - 0.5*dy, y[-1] + 0.5*dy

    col = ax.imshow(a.T, aspect='auto', extent=(x0, x1, y0, y1),
                    interpolation='nearest', origin='lower',
                    **kwargs)
    if colorbar:
        pl.colorbar(col)
    if pl.isinteractive():
        pl.show()

    return col

def shade_to_line(xvals, yvals, blend=1, ax=None, y0=0,
                  color='b'):
    """ Shade a region between two curves including a color gradient.
    
    Parameters
    ----------
    xvals, yvals : array_like
      Vertically shade to the line given by xvals, yvals
    y0 : array_like
      Start shading from these y values (default 0).
    blend : float (default 1)
      Start the cmap blending to white at this distance from `yvals`.
    color : mpl color
      Color used to generate the color gradient.

    Returns
    -------
    im : mpl image object
      object represeting the shaded region.
    """
    if ax is None:
        ax = pl.gca()

    import matplotlib as mpl

    yvals = np.asarray(yvals)
    xvals = np.asarray(xvals)
    y0 = np.atleast_1d(y0)
    if len(y0) == 1:
        y0 = np.ones_like(yvals) * y0[0]
    else:
        assert len(y0) == len(yvals)
        
    c = [color, '1']
    cm = mpl.colors.LinearSegmentedColormap.from_list('mycm', c)

    ymax = yvals.max()
    ymin = y0.min()
    X, Y = np.meshgrid(xvals, np.linspace(ymin, ymax, 1000))
    im = np.zeros_like(Y)
    for i in xrange(len(xvals)):
        cond = (Y[:, i] > yvals[i] - blend) & (Y[:, i] > y0[i])
        im[cond, i] = (Y[cond, i] - (yvals[i] - blend)) / blend
        cond = Y[:, i] > yvals[i]
        im[cond, i] = 1
        cond = Y[:, i] < y0[i]
        im[cond, i] = 0

    im = ax.imshow(im, extent=(xvals[0], xvals[-1], ymin, ymax),
                   origin='lower', cmap=cm, aspect='auto')
    return im


def shade_to_line_vert(yvals, xvals, blend=1, ax=None, x0=0,
                  color='b'):
    """ Shade a region between two curves including a color gradient.
    
    Parameters
    ----------
    yvals, xvals : array_like
      horizontally shade to the line given by xvals, yvals
    x0 : array_like
      Start shading from these x values (default 0).
    blend : float (default 1)
      Start the cmap blending to white at this distance from `yvals`.
    color : mpl color
      Color used to generate the color gradient.

    Returns
    -------
    im : mpl image object
      object represeting the shaded region.
    """
    if ax is None:
        ax = pl.gca()

    import matplotlib as mpl

    yvals = np.asarray(yvals)
    xvals = np.asarray(xvals)
    x0 = np.atleast_1d(x0)
    if len(x0) == 1:
        x0 = np.ones_like(xvals) * x0[0]
    else:
        assert len(x0) == len(xvals)
        
    c = [color, '1']
    cm = mpl.colors.LinearSegmentedColormap.from_list('mycm', c)

    xmax = xvals.max()
    xmin = x0.min()
    Y, X = np.meshgrid(yvals, np.linspace(xmin, xmax, 1000))
    im = np.zeros_like(X)
    for i in xrange(len(yvals)):
        cond = (X[:, i] > xvals[i] - blend) & (X[:, i] > x0[i])
        im[cond, i] = (X[cond, i] - (xvals[i] - blend)) / blend
        cond = X[:, i] > xvals[i]
        im[cond, i] = 1
        cond = X[:, i] < x0[i]
        im[cond, i] = 0

    art = ax.imshow(im.T, extent=(xmin, xmax, yvals[0], yvals[-1]),
                   origin='lower', cmap=cm, aspect='auto')
    return art, im, X, Y



def draw_arrows(x, y, ax=None, capsize=2,  ms=6, direction='up',
                c='k', **kwargs):
    """ Draw arrows that can be used to show limits.

    Extra keyword arguments are passed to `pyplot.scatter()`. To draw
    a shorter arrow, get the arrow length desired by reducing the `ms`
    value, then increase capsize until you are happy with the result,
    vice versa to draw a longer arrow.

    Parameters
    ----------
    x, y: float or arrays of shape (N,)
      x and y positions.
    direction: str {'up', 'down', 'left', 'right'}
      The direction in which the arrows should point.
    
    """
    arrowlength=10.
    capsize = min(capsize, arrowlength)
    yvert = np.array([0, arrowlength, arrowlength - capsize, arrowlength,
                      arrowlength - capsize, arrowlength])
    xvert = np.array([0, 0, 0.5*capsize, 0, -0.5*capsize, 0])

    if direction == 'down':
        arrow_verts = zip(xvert, -yvert)
    elif direction == 'up':
        arrow_verts = zip(xvert, yvert)
    elif direction == 'left':
        arrow_verts = zip(-yvert, xvert)
    elif direction == 'up':
        arrow_verts = zip(yvert, xvert)
    else:
        raise ValueError(
            "direction must be one of 'up', 'down', 'left', 'right'")

    if ax is None:
        pl.figure()
        ax = pl.gca()
    
    c = ax.scatter(x, y, s=(1000/6.)*ms, marker=None, verts=arrow_verts,
                   edgecolors=c, **kwargs)
    return c

