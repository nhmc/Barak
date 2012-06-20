""" Perform calculations on Spectral Energy Distributions (SEDs).

Based on the SED module in astLib by Matt Hilton, with some routines
copied from there (LGPL): http://astlib.sourceforge.net

- VEGA: The SED of Vega, used for calculation of magnitudes on the Vega system.
- AB: Flat spectrum SED, used for calculation of magnitudes on the AB system.
- SUN: The SED of the Sun.

"""
import os, math
import numpy as np
from convolve import convolve_psf
from io import readtabfits
import cosmology
from constants import c, c_kms, Jy
from utilities import get_data_path

try:
    import matplotlib.pyplot as pl
except:
    print "WARNING: failed to import matplotlib - some functions will not work."

datapath = get_data_path()
PATH_PASSBAND = datapath + '/passbands/'
PATH_EXTINCT = datapath + '/atmos_extinction/'
PATH_TEMPLATE = datapath + '/templates/'

def _listfiles(topdir):
    names = [n for n in os.listdir(topdir) if os.path.isdir(topdir + n)]
    files = dict([(n, []) for n in names])
    for name in sorted(names):
        for n in sorted(os.listdir(topdir + name)):
            if n != 'README' and not os.path.isdir(topdir + name + '/'+ n) and \
                   not n.startswith('effic') and \
                   not n.endswith('.py') and not n.endswith('.pdf'):
                files[name].append(n)
    return files

TEMPLATES = _listfiles(PATH_TEMPLATE)
PASSBANDS = _listfiles(PATH_PASSBAND)

def get_bands(instr=None, names=None, ccd=None):
    """ Get one or more passbands by giving the instrument and
    filename.

    If `names` is not given, then every passband for that instrument
    is returned.  Passband instruments and filenames are listed in the
    dictionary PASSBANDS. names can be a list, a single string, or a
    comma-separated string of values.

    Examples
    --------
    
    >>> sdss = get_bands('SDSS', 'u,g,r,i,z')  # get the SDSS passbands
    >>> U = get_bands('LBC', 'u')    # get the LBC U_spec filter
    """
    if instr is None:
        return _listfiles(PATH_PASSBAND)
    if isinstance(names, basestring):
        if ',' in names:
            names = [n.strip() for n in names.split(',')]
        else:
            return Passband(instr + '/' + names)
    elif names is None:
        names = PASSBANDS[instr]

    return [Passband(instr + '/' + n, ccd=ccd) for n in names]

def get_SEDs(kind=None, names=None):
    """ Get one or more SEDs based on their type and filename 

    If `names` is not given, then every SED of that type is returned.
    SED types and filenames are listed in the dictionary TEMPLATES.

    Examples
    --------
    >>> pickles = get_SEDs('pickles')   # pickles stellar library SEDs
    >>> lbga = get_SEDs('LBG', 'lbg_abs.dat')  # LBG absorption spectrum
    """
    if kind is None:
        return _listfiles(PATH_TEMPLATE)
    if isinstance(names, basestring):
        if ',' in names:
            names = [n.strip() for n in names.split(',')]
        else:
            return SED(kind + '/' + names)
    elif names is None:
        names = TEMPLATES[kind]

    return [SED(kind + '/' + n) for n in names]

def get_extinction(filename=None, airmass=1.):
    """ return the atmospheric extinction from the given file.

    returns extinction = 10^(0.4 * extinction_in_mags * airmass),

    where flux_true = flux_extincted * extinction
    """ 
    if filename is None:
        return sorted(os.listdir(PATH_EXTINCT))

    wa, emag = np.loadtxt(PATH_EXTINCT + filename, unpack=1)
    return wa, 10**(-0.4 * emag * airmass)


class Passband(object):
    """This class describes a filter transmission curve. Passband
    objects are created by loading data from from text files
    containing wavelength in angstroms in the first column, relative
    transmission efficiency in the second column (whitespace
    delimited). For example, to create a Passband object for the 2MASS
    J filter:
    
    passband = Passband('J_2MASS.res')

    where 'J_2MASS.res' is a file in the current working directory
    that describes the filter.

    The available passbands are in PASSBANDS.
    
    Attributes
    ----------
    wa : array of floats
      Wavelength in Angstroms
    tr : array of floats
      Normalised transmission, including atmospheric extinction and
      detector efficiency. May or may not include extinction from the
      optical path.
    effective_wa : float
      Effective wavelength of the passband.

    Methods
    -------
    plot
    """
    def __init__(self, filename, ccd=None):
        if not filename.startswith(PATH_PASSBAND):
            filepath = PATH_PASSBAND + filename
        self.wa, self.tr = np.loadtxt(filepath, usecols=(0,1), unpack=True)
        # check wavelengths are sorted lowest -> highest
        isort = self.wa.argsort()
        self.wa = self.wa[isort]
        self.tr = self.tr[isort]

        # get the name of th filter/passband file and the name of the
        # directory in which it lives (the instrument).
        prefix, filtername = os.path.split(filename)
        _, instr = os.path.split(prefix)
        self.name = filename

        if instr == 'LBC' and ccd is None:
            if filtername.startswith('LBCB') or filtername in 'ug':
                ccd = 'blue'
            elif filtername.startswith('LBCR') or filtername in 'riz':
                ccd = 'red'
        elif instr == 'FORS' and ccd is None:
            raise ValueError('Must specify detector (ccd=red or ccd=blue)')

        self.atmos = self.effic = None
        if ccd is not None:
            # apply ccd/optics efficiency
            name = PATH_PASSBAND + instr + '/effic_%s.txt' % ccd
            wa, effic = np.loadtxt(name, usecols=(0,1), unpack=1)
            self.effic = np.interp(self.wa, wa, effic)
            self.tr *= self.effic

        extinctmap = dict(LBC='kpno_atmos.dat', FORS='paranal_atmos.dat',
                          HawkI='paranal_atmos.dat',
                          KPNO_Mosaic='kpno_atmos.dat',
                          CTIO_Mosaic='ctio_atmos.dat')

        if instr in extinctmap:
            # apply atmospheric extinction
            wa, emag = np.loadtxt(PATH_EXTINCT + extinctmap[instr], unpack=1)
            self.atmos = np.interp(self.wa, wa, 10**(-0.4 * emag))
            self.tr *= self.atmos

        # trim away areas where band transmission is negligibly small
        # (<0.01% of peak transmission).
        isort = self.tr.argsort()
        sortedtr = self.tr[isort]
        maxtr = sortedtr[-1]
        imax = isort[-1]
        ind = isort[sortedtr < 1e-4 * maxtr]
        i,j = ind[ind < imax].max(), ind[ind > imax].min() 
        i = min(abs(i-2), 0)
        j += 1
        self.wa = self.wa[i:j]
        self.tr = self.tr[i:j]
        if self.atmos is not None:
            self.atmos = self.atmos[i:j]
        if self.effic is not None:
            self.effic = self.effic[i:j]
            
        # normalise
        self.ntr = self.tr / np.trapz(self.tr, self.wa)

        # Calculate the effective wavelength for the passband. This is
        # the same as equation (3) of Carter et al. 2009.
        a = np.trapz(self.tr * self.wa)
        b = np.trapz(self.tr / self.wa)
        self.effective_wa = math.sqrt(a / b)

        # find the AB and Vega magnitudes in this band for calculating
        # magnitudes.
        self.flux = {}
        self.flux['Vega'] = VEGA.calc_flux(self)
        self.flux['AB'] = AB.calc_flux(self)

    def __repr__(self):
        return 'Passband "%s"' % self.name

    def plot(self, effic=False, atmos=False, ymax=None, **kwargs):
        """ Plots the passband. We plot the non-normalised
        transmission. This may or may not include ccd efficiency,
        losses from the atmosphere and telescope optics.
        """
        tr = self.tr
        if ymax is not None:
            tr = self.tr / self.tr.max() * ymax
        pl.plot(self.wa, tr, **kwargs)
        if self.effic is not None and effic:
            pl.plot(self.wa, self.effic,
                    label='applied ccd efficiency', **kwargs)
        if self.atmos is not None and atmos:
            pl.plot(self.wa, self.atmos,
                    label='applied atmospheric extinction', **kwargs)

        pl.xlabel("Wavelength ($\AA$)")
        pl.ylabel("Transmission")
        if atmos or effic:
            pl.legend()
        if pl.isinteractive():
            pl.show()


class SED(object):
    """A Spectral Energy Distribution (SED).

    Instantiate with either a filename or a list of wavelengths and fluxes.
    Wavelengths must be in Angstroms, fluxes in erg/s/cm^2/Ang.

    To convert from f_nu to f_lambda in erg/s/cm^2/Ang, substitute
    using::

     nu = c / lambda
     f_lambda = c / lambda^2 * f_nu 

    Available SED templates filenames are in TEMPLATES.
    """    
    def __init__(self, filename=None, wa=[], fl=[], z=0., label=None):

        # filename overrides wave and flux keywords
        if filename is not None:
            if not filename.startswith(PATH_TEMPLATE):
                filepath = PATH_TEMPLATE + filename
            if filepath.endswith('.fits'):
                rec = readtabfits(filepath)
                wa, fl = rec.wa, rec.fl
            else:
                wa, fl = np.loadtxt(filepath, usecols=(0,1), unpack=1)  
            if label is None:
                label = filename
        # We keep a copy of the wavelength, flux at z = 0 as it's
        # more robust to copy that to self.wa, flux and
        # redshift it, rather than repeatedly redshifting
        self.z0wa = np.array(wa)
        self.z0fl = np.array(fl)
        self.wa = np.array(wa)
        self.fl = np.array(fl)
        self.z = z
        self.distance_modulus = 0
        self.label = label    

        # Store the intrinsic (i.e. unextincted) flux in case we
        # change extinction
        self.EBmV = 0.
        self.z0fl_no_extinct = np.array(fl)

        if abs(z) > 1e-6:
            self.redshift_to(z)


    def __repr__(self):
        return 'SED "%s"' % self.label

    def copy(self):
        """Copies the SED, returning a new SED object.
        """
        newSED = SED(wa=self.z0wa, fl=self.z0fl, z=self.z, label=self.label)
        return newSED

    def integrate(self, wmin=None, wmax=None):
        """ Calculates flux (erg/s/cm^2) in SED within given wavelength
        range."""
        if wmin is None:
            wmin = self.wa[0]
        if wmax is None:
            wmax = self.wa[-1]

        i,j = self.wa.searchsorted([wmin, wmax])
        fl = np.trapz(self.fl[i:j], self.wa[i:j])

        return fl

    def plot(self, log=False, ymax=None, **kwargs):
        fl = self.fl
        if ymax is not None:
            fl = self.fl / self.fl.max() * ymax

        label = '%s z=%.1f E(B-V)=%.2f' % (self.label, self.z, self.EBmV)
        if log:
            pl.loglog(self.wa, fl, label=label, **kwargs)
        else:
            pl.plot(self.wa, fl, label=label, **kwargs)
        pl.xlabel('Wavelength ($\AA$)')
        pl.ylabel('Flux (ergs s$^{-1}$cm$^{-2}$ $\AA^{-1}$)')
        #pl.legend()
        if pl.isinteractive():
            pl.show()

    def redshift_to(self, z, cosmo=None):
        """Redshifts the SED to redshift z. """        
        # We have to conserve energy so the area under the redshifted
        # SED has to be equal to the area under the unredshifted SED,
        # otherwise magnitude calculations will be wrong when
        # comparing SEDs at different zs

        self.wa = np.array(self.z0wa)
        self.fl = np.array(self.z0fl)
        
        z0fluxtot = np.trapz(self.z0wa, self.z0fl)
        self.wa *= z + 1
        zfluxtot = np.trapz(self.wa, self.fl)
        self.fl *= z0fluxtot / zfluxtot
        self.z = z
        if z > 0:
            if cosmo is None:
                cosmo = cosmology.get_default()
            self.distance_modulus = cosmo.distmod(z)
        else:
            self.distance_modulus = 0.

    def normalise_to_mag(self, ABmag, band):
        """Normalises the SED to match the flux equivalent to the
        given AB magnitude in the given passband.
        """
        magflux, err = mag2flux(ABmag, 0., band)
        sedflux = self.calc_flux(band)
        norm = magflux / sedflux
        self.fl *= norm
        self.z0fl *= norm
        
    def calc_flux(self, band):
        """Calculates flux in the given passband. """
        if self.wa[0] > band.wa[0] or self.wa[-1] < band.wa[-1]:
            msg = "SED does not cover the whole bandpass, we're extrapolating"
            print 'WARNING:', msg
            dw = np.median(np.diff(self.wa))
            sedwa = np.arange(band.wa[0], band.wa[-1]+dw, dw)
            sedfl = np.interp(sedwa, self.wa, self.fl)
        else:
            sedwa = self.wa
            sedfl = self.fl

        i,j = sedwa.searchsorted([band.wa[0], band.wa[-1]])
        fl = sedfl[i:j]
        wa = sedwa[i:j]

        dw_band = np.median(np.diff(band.wa))
        dw_sed = np.median(np.diff(wa))
        if dw_sed > dw_band and dw_band > 20:
            print ('WARNING: SED wavelength sampling interval ~%.2f Ang, '
                   'but bandpass sampling interval ~%.2f Ang' % (dw1, dw0)) 
            
        # interpolate the band normalised transmission to the SED
        # wavelength values.
        band_ntr = np.interp(wa, band.wa, band.ntr)
        sed_in_band =  band_ntr * fl
        flux = np.trapz(sed_in_band * wa, wa) / np.trapz(band_ntr * wa, wa) 
        return flux

    def calc_mag(self, band, system="Vega"):
        """Calculates magnitude in the given passband.

        Note that the distance modulus is not added.

        `system` is either 'Vega' or 'AB'
        """
        f1 = self.calc_flux(band)

        if f1 > 0:
            mag = -2.5 * math.log10(f1/band.flux[system])
            # Add 0.026 because Vega has V=0.026 (e.g. Bohlin & Gilliland 2004)
            if system == "Vega":
                mag += 0.026        
            #mag += self.distance_modulus
        else:
            mag = np.inf

        return mag
    
    def calc_colour(self, band1, band2, system="Vega"):
        """Calculates the colour band1 - band2.

        system is either 'Vega' or 'AB'.
        """
        mag1 = self.calc_mag(band1, system=system)
        mag2 = self.calc_mag(band2, system=system)
    
        return mag1 - mag2

    def apply_extinction(self, EBmV):
        """Applies the Calzetti et al. 2000 (ApJ, 533, 682) extinction
        law to the SED with the given E(B-V) amount of extinction.
        R_v' = 4.05 is assumed (see equation (5) of Calzetti et al.).

        Call with EBmV = 0 to remove any previously-applied extinction.
        """

        # All done in rest frame
        self.z0fl[:] = self.z0fl_no_extinct

        # Allow us to set EBmV == 0 to turn extinction off
        if EBmV > 1e-10:
            reddening = calc_extinction_Calzetti(self.z0wa, EBmV)
            self.z0fl /= reddening
            self.EBmV = EBmV
        else:
            self.EBmV = 0

        self.redshift_to(self.z)


def mag2flux(ABmag, ABmagerr, band):
    """Converts given AB magnitude and uncertainty into flux in the
    given band, in erg/s/cm^2/Angstrom.

    Returns the flux and error in the given band.
    """
    # AB mag (See Oke, J.B. 1974, ApJS, 27, 21)
    flux_nu = 10**(-(ABmag + 48.6)/2.5)          # erg/s/cm^2/Hz
    # for conversion to erg s-1 cm-2 angstrom-1 with lambda in microns
    lam_eff = band.effective_wa * 1e-8           # cm
    flux_lam = 1e-8 * c * flux_nu / lam_eff**2   # erg/s/cm^2/Ang

    flux_nu_err =  10**(-(ABmag - ABmagerr + 48.6)/2.5)
    flux_lam_err = 1e-8 * c * flux_nu_err / lam_eff**2  # erg/s/cm^2/Ang
    flux_lam_err = flux_lam_err - flux_lam

    return flux_lam, flux_lam_err

def flux2mag(flux, fluxerr, band):
    """Converts given flux and uncertainty in erg/s/cm^2/Angstrom into
    AB magnitudes.

    Returns the magnitude and error in the given band.
    """
    lam_eff = band.effective_wa * 1e-8          # cm
    flux_wa = 1e8 * flux                        # erg/s/cm^2/cm
    flux_nu = flux_wa * lam_eff**2 / c          # erg/s/cm^2/Hz
    mag = -2.5*math.log10(flux_nu) - 48.6

    flux_nuerr = 1e8 * fluxerr * lam_eff**2 / c
    magerr = mag - (-2.5*math.log10(flux_nu + flux_nuerr) - 48.6)
    
    return mag, magerr

def mag2Jy(ABmag):
    """Converts an AB magnitude into flux density in Jy.
    """
    flux_nu = 10**(-(ABmag + 48.6)/2.5) / Jy
    return flux_nu

def Jy2Mag(fluxJy):
    """Converts flux density in Jy into AB magnitude.
    """
    ABmag = -2.5 * (np.log10(fluxJy * Jy)) - 48.6
    return ABmag

def calc_extinction_Calzetti(wa, EBmV):
    """ Find the extinction as a function of wavelength for the given
    E(B-V) using the relation from Calzetti et al. 2000 (ApJ, 533,
    682).  R_v' = 4.05 is assumed (see equation (5) of Calzetti et
    al.) E(B-V) is the extinction in the stellar continuum.

    The wavelength array wa must be in Angstroms and sorted lowest ->
    highest.

    Returns reddening such that::

     extincted_flux = unextincted_flux * reddening
    """
    assert wa[0] < 22000 and wa[-1] > 1200

    # Note that EBmV is assumed to be Es as in equations (2) - (5)
    
    # equation (5) of Calzetti et al. 2000
    Rv = 4.05

    # Constants belwo assume wavelength is in microns.
    
    uwa = np.array(wa / 10000.)
    k = np.zeros_like(wa)

    def kshort(wa):
        return 2.659*(-2.156 + (1.509 - 0.198/wa + 0.011/wa**2)/wa) + Rv

    def klong(wa):
        return 2.659*(-1.857 + 1.040/wa) + Rv

    # populate k in a piece-wise fashion, extrapolating
    # below 1200 and above 22000 Angstroms
    if uwa[0] < 0.12:
        slope = (kshort(0.11) - kshort(0.12)) / (0.11 - 0.12)
        intercept = kshort(0.12) - slope * 0.12
        i = uwa.searchsorted(0.12)
        k[:i] = slope * uwa[:i] + intercept
    if uwa[0] < 0.63 and uwa[-1] > 0.12:
        i,j = uwa.searchsorted([0.12, 0.63])
        k[i:j] = kshort(uwa[i:j])
    if uwa[0] < 2.2 and uwa[-1] > 0.63:
        i,j = uwa.searchsorted([0.63, 2.2])
        k[i:j] = klong(uwa[i:j])
    if uwa[-1] >= 2.2:
        slope = (klong(2.19) - klong(2.2)) / (2.19 - 2.2)
        intercept = klong(2.19) - slope * 2.19
        i = uwa.searchsorted(2.2)
        k[i:] = slope * uwa[i:] + intercept

    # fudge to stop going negative
    k[k <= 1e-6] = 1e-6
        
    reddening = 10**(0.4 * EBmV * k)
    return reddening


# Data
VEGA = SED('reference/Vega_bohlin2006')
SUN = SED('reference/sun_stis')


wa = np.logspace(1, 10, 1e5)
# AB SED has constant flux density 3631 Jy
fl = 1e-8 * 3631 * Jy * c / (wa * 1e-8)**2  # erg/s/cm^2/Ang
AB = SED(wa=wa, fl=fl)
# don't clutter the namespace
del wa, fl
