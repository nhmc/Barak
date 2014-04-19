""" Plotting routines. """

# python 2.6+ compatibility
from __future__ import division, print_function, unicode_literals
try:
    unicode
except NameError:
    unicode = basestring = str
    xrange = range

from math import log10
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
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
        levels.append(np.percentile(d, percentiles))
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
    ax.plot(x, levels[:,2], '-', color=color, **kwargs)
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

    return [l]

def hist_yedge(y, ax, bins=20, height=0.2, histmax=None, fmt='',
               loc='right', **kwargs):
    y, ybins = np.histogram(y, bins=bins)

    b = np.repeat(ybins, 2)
    Y = np.concatenate([[0], np.repeat(y,2), [0]])
    Ymax = (histmax if histmax is not None else y.max())
    Y = height * Y / Ymax
    if 'right' in loc:
        Y = 1 - Y
    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    artist, = ax.plot(Y, b, fmt, transform=trans, **kwargs)
    
    return artist

def hist_xedge(x, ax, bins=20, height=0.2, histmax=None, fmt='',
               loc='bottom', **kwargs):
    x, xbins = np.histogram(x, bins=bins)
    b = np.repeat(xbins, 2)
    X = np.concatenate([[0], np.repeat(x,2), [0]])
    Xmax = (histmax if histmax is not None else x.max())
    X = height * X / Xmax
    if 'top' in loc:
        X = 1 - X
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    artist, = ax.plot(b, X, fmt, transform=trans, **kwargs)
    return artist

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

    a = np.asarray(a).ravel()
    a = a[~np.isnan(a)]
    vals,bins = np.histogram(a, bins=bins)
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
    x : shape (M,)
      Coordinates, must be equally spaced.
    y : shape (N,)
      Coordinates, must be equally spaced.
    ax : axes
      Axes in which to plot.
    colorbar : bool (True)
      Whether to also plot a colorbar.
    """
    if x is None:
        x = np.arange(a.shape[1])
    else:
        assert len(x) == a.shape[1]
    
    if y is None:
        y = np.arange(a.shape[0])
    else:
        assert len(y) == a.shape[0]

    if ax is None:
        pl.figure()
        ax = pl.gca()

    dxvals = x[1:] - x[:-1]
    dx = dxvals[0]
    assert dx > 0
    assert np.allclose(dx, dxvals[1:])
    x0, x1 = x[0] - 0.5*dx, x[-1] + 0.5*dx

    dyvals = y[1:] - y[:-1]
    dy = dyvals[0]
    assert dy > 0
    assert np.allclose(dy, dyvals[1:])
    y0, y1 = y[0] - 0.5*dy, y[-1] + 0.5*dy

    col = ax.imshow(a, aspect='auto', extent=(x0, x1, y0, y1),
                    interpolation='nearest', origin='lower',
                    **kwargs)
    if colorbar:
        pl.colorbar(col)
    if pl.isinteractive():
        pl.show()

    return col

def shade_to_line(xvals, yvals, blend=1, ax=None, y0=0, color='b'):
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
    al = arrowlength
    yverts = (np.array([al, al-capsize]),
              np.array([al, 0]),
              np.array([al, al-capsize]))
    xverts = (np.array([0, 0.5*capsize]),
              np.array([0, 0]),
              np.array([0, -0.5*capsize]))

    if ax is None:
        pl.figure()
        ax = pl.gca()

    art = []
    for xvert,yvert in zip(xverts, yverts):
        if direction == 'down':
            arrow_verts = list(zip(xvert, -yvert))
        elif direction == 'up':
            arrow_verts = list(zip(xvert, yvert))
        elif direction == 'left':
            arrow_verts = list(zip(-yvert, xvert))
        elif direction == 'up':
            arrow_verts = list(zip(yvert, xvert))
        else:
            raise ValueError(
                "direction must be one of 'up', 'down', 'left', 'right'")

        art.append(ax.scatter(x, y, s=(1000/6.) * ms, marker=None,
                            verts=arrow_verts, edgecolors=c, **kwargs))
    return art

def calc_log_minor_ticks(majorticks):
    """ Get minor tick positions for a log scale.

    Parameters
    ----------
    majorticks : array_like
        log10 of the major tick positions.

    Returns
    -------
    minorticks : ndarray
        log10 of the minor tick positions.
    """
    tickpos = np.log10(np.arange(2, 10))
    minorticks = []
    for t in np.atleast_1d(majorticks):
        minorticks.extend(t + tickpos)

    return minorticks

def make_log_xlabels(ax, yoff=-0.05):
    """ Make the labels on the x axis log.
    """
    x0, x1 = ax.get_xlim()
    ticks = ax.get_xticks()
    tdecade = [t for t in ticks if np.allclose(t % 1, 0)]
    floor = np.floor(ticks[0])
    ceil = np.ceil(ticks[-1])
    if tdecade[0] > floor:
        tdecade = [floor] + tdecade 
    elif tdecade[0] < ceil:
        tdecade.append(ceil) 
    minorticks = calc_log_minor_ticks(tdecade)
    ax.set_xticks(minorticks, minor=True)
    ax.set_xticks(tdecade)
    ticklabels = []
    for t in tdecade:
        if t == 0:
            ticklabels.append('$\mathdefault{1}$')
        elif t == 1:
            ticklabels.append('$\mathdefault{10}$')
        elif t == -1:
            ticklabels.append('$\mathdefault{0.1}$')
        elif t == 2:
            ticklabels.append('$\mathdefault{100}$')
        elif t == -2:
            ticklabels.append('$\mathdefault{0.01}$')
        else:
            ticklabels.append('$\mathdefault{10^{%.0f}}$' % t)


    ax.set_xticklabels(ticklabels)
    for t in ax.xaxis.get_ticklabels():
        t.set_verticalalignment('bottom')
        t.set_y(yoff)

    ax.set_xlim(x0, x1)
    return ax

def make_log_ylabels(ax):
    """ make the labels on the y axis log.
    """
    y0, y1 = ax.get_ylim()
    ticks = ax.get_yticks()
    tdecade = [t for t in ticks if np.allclose(t % 1, 0)]
    floor = np.floor(ticks[0])
    ceil = np.ceil(ticks[-1])
    if tdecade[0] > floor:
        tdecade = [floor] + tdecade 
    elif tdecade[0] < ceil:
        tdecade.append(ceil) 
    minorticks = calc_log_minor_ticks(tdecade)
    ax.set_yticks(minorticks, minor=True)
    ax.set_yticks(tdecade)
    ticklabels = []
    for t in tdecade:
        if t == 0:
            ticklabels.append('$\mathdefault{1}$')
        elif t == 1:
            ticklabels.append('$\mathdefault{10}$')
        elif t == -1:
            ticklabels.append('$\mathdefault{0.1}$')
        elif t == 2:
            ticklabels.append('$\mathdefault{100}$')
        elif t == -2:
            ticklabels.append('$\mathdefault{0.01}$')
        else:
            ticklabels.append('$\mathdefault{10^{%.0f}}$' % t)

    ax.set_yticklabels(ticklabels)
    ax.set_ylim(y0, y1)
    return ax


def plot_ticks_wa(ax, wa, fl, height, ticks, keeponly=None, labels=True,
                  c='k'):
    """ plot a ticks on a wavelength scale.

    This plots ticks (such as those returned by `find_tau()`) on a
    spectrum.

    Parameters
    ----------
    ax : matplotlib axes
      The axes on which to plot the ticks.
    wa, fl : array_like
      wavelength and flux of spectrum. `wa` must be sorted.
    height : float
      tick height in flux units.
    ticks : record array
      A record array of the sort returned by `find_tau`. The fields
      `wa`, `wa0`, and `name` are required.
    keeponly : str
      If this is not None (the default), then only plot ticks that
      contain this string in their name.
    labels : bool (True)
      Whether to plot labels next to the tickmarks.
    c : matplotlib colour ('k')
      Tick colour. The default is black.

    Returns
    -------
    Ticks, Tlabels : Matplotlib collection of tickmarks and tick labels.
      The artists corresponding to the ticks and their labels.
    """
    ind = wa.searchsorted(ticks.wa)
    c0 = (ind == 0) | (ind == len(wa))
    ticks = ticks[~c0]
    ymin = fl[ind[~c0]]*1.1
    Tlabels = []
    c1 = np.ones(len(ticks), bool)
    for i,t in enumerate(ticks):
        if keeponly is not None:
            if keeponly not in t.name:
                c1[i] = False
                continue
        if not labels:
            continue
        label = '%s %.0f' % (t.name, t.wa0)
        label = label.replace('NeVII', 'NeVIII')
        Tlabels.append(ax.text(t.wa, ymin[i] + 1.1*height, label, rotation=60,
                               fontsize=8, va='bottom', alpha=0.7))

    Ticks = ax.vlines(ticks.wa[c1], ymin[c1], ymin[c1] + height, color=c,
                      lw=1)

    return Ticks, Tlabels

def get_subplot(nrow, ncol, num):
    """ Get a matplotlib subplot.

    Like pylab.subplot, but the numbering goes down columns rather
    than across rows.
    """
    i = num - 1
    num_new = (i % nrow ) * ncol + i // nrow + 1
    return plt.subplot(nrow, ncol, num_new)


def makefig(cols, rows, width=8, height=None, left=1.5, bottom=1.5, top=1.5,
              right=0.7, horizgap=1, vertgap=1):
    """ Make a figure that will hold a set of plots with a fixed size
    in cm.
    """
    W = float(width)*cols + float(horizgap)*(cols-1) + left + right
    height = height or width
    H = float(height)*rows + float(vertgap)*(rows-1) + top + bottom
    fig = pl.figure(figsize=(W/2.54, H/2.54))
    fig.subplots_adjust(left=left/W, bottom=bottom/H,
                        wspace=horizgap/float(width),
                        hspace=vertgap/float(height),
                        right=1 - right/W, top=1 - top/H)
    return fig

def subplots(cols, rows, *args, **kwargs):
    """ Create an axes that covers a portion of the figure.

    The axes has position and extent given by a grid of rows and cols
    and the pos string, which uses indexing syntax. If a fig keyword
    is not present giving the figure, a figure is generated using
    makefig. keywords can be passed to makefig; the defaults are
    width=8, left=1.5, bottom=1.5, top=1.5, right=0.7, horizgap=1 and
    vertgap=1.

    If the indexing argument is omitted, then a grid of plots is
    returned, from upper left to lower right ordered such that columns
    are increasing fastest.

    Inspired by the Supermongo subplots command.

    Examples
    --------
    >>> axleft, axright = subplots(2, 1)

    >>> ax = subplots(3, 3, '1:,2')
    >>> pylab.show()

    More examples of indexing notation

    >>> ax = subplots(3, 3, '1,0:2')
    >>> ax = subplots(3, 3, '2,0')
    >>> ax = subplots(3, 3, '2,1')

    Use a single call to generate multiple subplotss

    >>> ax1,ax2,ax3 = subplots(3, 3, '0,: 1:,:2 1:,2')
    """
    fig = kwargs.pop('fig', None)
    if fig is None:
        fig = makefig(cols, rows, **kwargs)

    pars = fig.subplotpars
    # in units of figure fraction
    w = (pars.right - pars.left) / (cols + pars.wspace * (cols-1))
    h = (pars.top - pars.bottom) / (rows + pars.hspace * (rows-1))

    xgap = pars.wspace * w
    ygap = pars.hspace * h

    if len(args) == 0:
        axes = [fig.add_subplot(rows,cols, i+1) for i in range(rows*cols)]
        return tuple(axes)
    elif len(args) > 1:
        raise "Only a single position string is allowed"

    axes = []
    for pos in args[0].split():
        ix,iy = pos.split(',')
        temp = []
        for i,ind in enumerate([ix,iy]):
            if i == 0:
                ntot = cols
            else:
                ntot = rows
            start = 0
            if ':' in ind:
                if ind == ':':
                    span = ntot
                elif ind.startswith(':'):
                    span = int(ind[1:])
                elif ind.endswith(':'):
                    start = int(ind[:-1])
                    span = ntot - start
                else:
                    start,end = map(int, ind.split(':'))
                    span = min(end - start, ntot)
            else:
                start = int(ind)
                span = 1
            temp.append((start, span))

        (i,ispan),(j,jspan) = temp
        left1 = pars.left + i*(w + xgap)
        bottom1 = pars.bottom + j*(h + ygap)
        w1 = ispan*w + (ispan-1)*xgap
        h1 = jspan*h + (jspan-1)*ygap

        axes.append(fig.add_axes([left1, bottom1, w1, h1]))

    if len(axes) == 1:
        return axes[0]
    else:
        return tuple(axes)

def get_fig_axes(nrows, ncols, nplots, width=11.7, height=None, aspect=0.5):
    """ Generate a figure with a number of equally-sized subplots.

    Parameters
    ----------
    nrows : int
      Number of rows
    ncols : int
      Number of columns
    nplots : int
      Number of plots in total
    width : float
      Width of the figure in inches
    aspect : float
      width / height of each sub-plot.

    Returns
    -------
    fig : matplotlib figure object
    ax : dictionary with matplotlib axes objects
      ordered from top left going down columns.
    """
    if height is None:
        height = width*aspect*nrows/ncols
    fig = pl.figure(figsize=(width, height))

    axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nplots)]

    # reorder axes so they go from top left down columns instead of across
    axes1 = []
    ileft = []
    ibottom = []
    for j in range(ncols):
        for i in range(nrows):
            ind = j + i*ncols
            if ind > nplots - 1:
                continue
            axes1.append(axes[ind])

    # find the indices of the left and bottom plots (used to set axes
    # labels)
    ileft = range(nrows)
    ibottom = [i*nrows - 1 for i in range(1, ncols+1)]
    for i in range(ncols*nrows - nplots):
        ibottom[-(i+1)] -= ncols*nrows - nplots - i

    ax = dict(axes=axes1, nrows=nrows, ncols=ncols,
              ileft=ileft, ibottom=ibottom)
    return fig, ax

def get_nrows_ncols(nplots, prefer_rows=True):
    """ Get the number of rows and columns to plot a given number of plots.

    Parameters
    ----------
    nplots : int
      Desired number of plots.

    Returns
    -------
    nrows, ncols : int
    """
    nrows = max(int(np.sqrt(nplots)), 1)
    ncols = nrows
    while nplots > (nrows * ncols):
        if prefer_rows:
            nrows += 1
        else:
            ncols += 1

    return nrows, ncols

def get_flux_plotrange(fl):
    ymax = abs(np.percentile(fl[~np.isnan(fl)], 95)) * 1.5
    return -0.1*ymax, ymax
