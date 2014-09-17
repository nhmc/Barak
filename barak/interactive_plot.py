""" Classes useful for making interactive plots.
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from barak.utilities import between
from barak.plot import get_flux_plotrange
from barak.convolve import convolve_psf
from barak.interp import AkimaSpline
from barak.io import savejson

class PlotWrapBase(object):
    """ A base class that has all the navigation and smoothing
    keypress events. These need to be connected to the wrapped figure
    of the class inheriting from this with something like:

    ** You need to define the following attributes **

    self.wa, self.fl    Spectrum wavelength and flux
    self.nsmooth        integer > 0 that determines the smoothing
    self.ax             Axes where specturm is plotted
    self.fig            Figure which holds the axes.
    self.artists['fl']  The Matplotlib line artist that represents the flux.

    def connect(self, fig):
        cids = dict(key=[])
        # the top two are methods of PlotWrapBase
        cids['key'].append(fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_navigate))
        cids['key'].append(fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_smooth))
        self.cids.update(cids)
    """
    _help_string = """
i,o          Zoom in/out x limits
y            Zoom out y limits
Y            Guess y limits
t,b          Set y top/bottom limit
l,r          Set left/right x limit
[,]          pan left/right
w            Return to original view

S,U          Smooth/unsmooth spectrum
"""

    def __init__(self):
        pass

    def on_keypress_navigate(self, event):
        """ Process a keypress event. Requires attributes self.ax,
        self.fl, self.wa, self.fig
        """
        # Navigation
        if event.key == 'i' and event.inaxes:
            x0,x1 = self.ax.get_xlim()
            x = event.xdata
            dx = abs(x1 - x0)
            self.ax.set_xlim(x - 0.275*dx, x + 0.275*dx)
            self.fig.canvas.draw()
        elif event.key == 'o' and event.inaxes:
            x0,x1 = self.ax.get_xlim()
            x = event.xdata
            dx = abs(x1 - x0)
            self.ax.set_xlim(x - 0.95*dx, x + 0.95*dx)
            self.fig.canvas.draw()
        elif event.key == 'Y' and event.inaxes:
            y0,y1 = self.ax.get_ylim()
            y = event.ydata
            dy = abs(y1 - y0)
            self.ax.set_ylim(y0 - 0.05*dy, y1 + 0.4*dy)
            self.fig.canvas.draw()
        elif event.key == 'y' and event.inaxes:
            x0,x1 = self.ax.get_xlim()            
            y0,y1 = get_flux_plotrange(self.fl[between(self.wa, x0, x1)])
            self.ax.set_ylim(y0, y1)
            self.fig.canvas.draw()
        elif event.key == ']':
            x0,x1 = self.ax.get_xlim()
            dx = abs(x1 - x0)
            self.ax.set_xlim(x1 - 0.1*dx, x1 + 0.9*dx)
            self.fig.canvas.draw()
        elif event.key == '[':
            x0,x1 = self.ax.get_xlim()
            dx = abs(x1 - x0)
            self.ax.set_xlim(x0 - 0.9*dx, x0 + 0.1*dx)
            self.fig.canvas.draw()
        elif event.key == 'w':
            self.ax.set_xlim(self.wa[0], self.wa[-1])
            y0,y1 = get_flux_plotrange(self.fl)
            self.ax.set_ylim(y0, y1)
            self.fig.canvas.draw()
        elif event.key == 'b' and event.inaxes:
            y0, y1 = self.ax.get_ylim()
            self.ax.set_ylim(event.ydata, y1)
            self.fig.canvas.draw()
        elif event.key == 't' and event.inaxes:
            y0, y1 = self.ax.get_ylim()
            self.ax.set_ylim(y0, event.ydata)
            self.fig.canvas.draw()
        elif event.key == 'l' and event.inaxes:
            x0, x1 = self.ax.get_xlim()
            self.ax.set_xlim(event.xdata, x1)
            self.fig.canvas.draw()
        elif event.key == 'r' and event.inaxes:
            x0, x1 = self.ax.get_xlim()
            self.ax.set_xlim(x0, event.xdata)
            self.fig.canvas.draw()

    def on_keypress_smooth(self, event):
        """ Smooth the flux with a gaussian. Requires attributes
        self.fl and self.nsmooth, self.artists['fl'] and self.fig."""
        if event.key == 'S':
            if self.nsmooth > 0:
                self.nsmooth += 0.5
            else:
                self.nsmooth = 1
            sfl = convolve_psf(self.fl, self.nsmooth)
            self.artists['fl'].set_ydata(sfl)
            self.fig.canvas.draw()
        elif event.key == 'U':
            self.nsmooth = 0
            self.artists['fl'].set_ydata(self.fl)
            self.fig.canvas.draw()


class PlotWrapBase_Continuum(PlotWrapBase):
    """
    This needs the following attributes defined:

    self.contpoints: List of x,y pairs giving the spline knots
    self.co:         Continuum array
    self.wa:         Wavelength array (sorted low to high)
    self.fl:         Flux array
    self.artists['contpoints']: Matplotlib artist showing the spline knots.
    self.artists['co']:  Matplotlib artist showing the continuum.

    It will also update a self.artists['model'] line using the new
    continuum and the array in self.model, if they exist.

    The knots are saved in the current directory in a file call
    '_knots.json'. This save location and name can be changed using
    the attributes self.outdir and self.name.
    """

    _help_string = PlotWrapBase._help_string + """
3,4          Add (3) or delete (4) a continuum point
"""
    def __init__(self):
        PlotWrapBase.__init__(self)

    def update_cont(self):
        """ Sort the continuum points and update continuum artists. 
        """
        if not self.contpoints:
            self.artists['contpoints'].set_data([0], [0])
        else:
            self.contpoints = sorted(self.contpoints, key=lambda pt: pt[0])
            cpts = self.contpoints
            self.artists['contpoints'].set_data(list(zip(*cpts)))

            name = '_knots.json'
            if hasattr(self, 'outdir') and hasattr(self, 'name'):
                name = self.outdir + self.name + '_knots.json'
            savejson(name, cpts, overwrite=True)
            if self.co is None:
                self.co = np.empty_like(self.fl)
            if len(cpts) > 2:
                i,j = self.wa.searchsorted([cpts[0][0], cpts[-1][0]])
                spline = AkimaSpline(*list(zip(*cpts)))
                self.co[i:j] = spline(self.wa[i:j])
                self.co[:i] = self.co[i]
                self.co[j:] = self.co[j-1]
            elif len(cpts) > 1:
                wa_pts, pts = zip(*cpts)
                self.co = np.interp(self.wa, wa_pts, pts)
            elif len(cpts) == 1:
                self.co[:] = cpts[0][1]

        if hasattr(self, 'model') and len(self.co) == len(self.model):
            self.artists['model'].set_data(self.wa, self.co * self.model)
        self.artists['co'].set_data(self.wa, self.co)

    def on_keypress_continuum(self, event):
        """ Add and remove continuum points.

        needs

        self.contpoints
        self.fig
        self.ax
        """
        if event.key == '3' and event.inaxes:
            # add a point to contpoints
            x,y = event.xdata, event.ydata
            if not self.contpoints or x not in zip(*self.contpoints)[0]:
                self.contpoints.append((x, y))                
                self.update_cont()
            self.fig.canvas.draw()
        elif event.key == '4' and event.inaxes:
            # remove a point from contpoints
            if not self.contpoints:
                'Define a continuum point first!'
            elif len(self.contpoints) == 1:
                self.contpoints = []
            else:
                contx,conty = zip(*self.ax.transData.transform(self.contpoints))
                sep = np.hypot(event.x - np.array(contx),
                               event.y - np.array(conty))
                self.contpoints.remove(self.contpoints[sep.argmin()])
            self.update_cont()
            self.fig.canvas.draw()

