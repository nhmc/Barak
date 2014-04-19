import numpy as np
from barak.utilities import between
from barak.convolve import convolve_psf


class PlotWrapBase(object):
    """ A base class that has all the navigation and smoothing
    keypress events. These need to be connected to the wrapped figure
    of the class inheriting from this with something like:

    def connect(self, fig):
        cids = dict(key=[])
        # the top two are methods of PlotWrapBase
        cids['key'].append(fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_navigate))
        cids['key'].append(fig.canvas.mpl_connect(
            'key_press_event', self.on_keypress_smooth))
        self.cids.update(cids)
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
