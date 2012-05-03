""" This module has routines for analysing the absorption profiles
from ions and molecules.
"""
import math
from math import pi, sqrt

import numpy as np
import voigt
from convolve import convolve_psf
from utilities import between, adict, get_data_path, indexnear
from constants import Ar, me, mp, k, c, e, sqrt_ln2, c_kms

def calctau(v, wav0, osc, gam, logN, T=None, btemp=20, bturb=0,
            debug=False, verbose=True):
    """ Returns the optical depth (Voigt profile) for a transition.

    Given an transition with rest wavelength wav0, osc strength,
    natural linewidth gam; b parameter (doppler and turbulent); and
    log10 (column density), returns the optical depth in velocity
    space. v is an array of velocity values in km/s. The absorption
    line must be centred at v=0.

    Parameters
    ----------
    v : array of floats, shape (N,)
      Velocities in km/s.
    wav0 : float
      Rest wavelength op transition in Angstroms.
    osc : float
      Oscillator strength of transition (dimensionless).
    gam : float
      Gamma parameter for the transition (dimensionless).
    logN : float:
      log10 of the column density in absorbers per cm^2.
    btemp : float (20)
      *b* parameter from doppler temperature broadening (km/s)
    bturb : float (0)
      *b* parameter from doppler turbulent broadening (km/s)
    T : float (None)
      Temperature of cloud in Kelvin (overrides btemp).

    Returns
    -------
    tau : array of floats, shape (N,)
      The optical depth as a function of `v`.

    Notes
    -----
    The step size for `v` must be small enough to properly
    sample the profile.

    To map the velocity array to some wavelength for a transitions
    with rest wavelength wav0 at redshift z:

    >>> z = 3.0
    >>> wa = wav0 * (1 + z) * (1 + v/c_kms)
    """
    # note units are cgs
    wav0 = wav0 * 1e-8                    # cm
    N = 10**logN                          # absorbers/cm^2
    if T is not None:
        btemp = sqrt(2*k*T / mp) * 1e-5     # km/s
    b = math.hypot(btemp, bturb) * 1e5    # cm/s
    nu0 = c / wav0                        # rest frequency, s^-1
    # Now use doppler relation between v and nu assuming gam << nu0
    gam_v = gam / nu0 * c             # cm/s
    if debug:
        print ('Widths in km/s (Lorentzian Gamma, Gaussian b):',
               gam_v/1.e5, b/1.e5)

    fwhml = gam_v / (2.*pi)                # cm/s
    fwhmg = 2. * sqrt_ln2 * b                # cm/s

    ##### sampling check, first get velocity width ######
    ic = np.searchsorted(v, 0)
    try:
        # it's ok if the transition is outside the fitting region.
        if ic == len(v):
            ic -= 1
        if ic == 0:
            ic += 1
        vstep = v[ic] - v[ic-1]
    except IndexError:
        raise IndexError(4*'%s ' % (len(v), ic, ic-1, v))

    fwhm = max(gam_v/1.e5, fwhmg/1.e5)

    if verbose and vstep > fwhm:
        print 'Warning: tau profile undersampled!'
        print '  Pixel width: %f km/s, transition fwhm: %f km/s' % (vstep,fwhm)
        # best not to correct for this here, because even if we do,
        # we'll get nonsense if we convolve the resulting flux with an
        # instrumental profile.  Need to use smaller dv size
        # throughout tau, exp(-tau) and convolution of exp(-tau)
        # calculations, only re-binning back to original dv size after
        # all these steps.

    u = 1.e5 / b * v                      # dimensionless
    a = gam_v / (4*pi*b)                       # dimensionless
    vp = voigt.voigt(a, u)                     # dimensionless
    const = pi * e**2 / (me*c)                 # m^2/s
    tau = const*N*osc*wav0 / (sqrt(pi)*b) * vp # dimensionless

    return tau

def calc_iontau(wa, ion, zp1, logN, b, debug=False, ticks=False, maxdv=1000.,
                label_tau_threshold=0.01, vpad=500., verbose=True):
    """ Returns tau values at each wavelength for transitions in ion.

    Parameters
    ----------
    wa : array of floats
      wavelength array
    ion : atom.data entry
      ion entry from readatom output dictionary
    zp1 : float
      redshift + 1
    logN : float
      log10(column density in cm**-2)
    b : float
      b parameter (km/s).  Assumes thermal broadening.
    maxdv : float (default 1000)
      For performance reasons, only calculate the Voigt profile for a
      single line to +/- maxdv.  Increase this if you expect DLA-type
      extended wings. None for no maximum.
    vpad : float (default 500)
      Include transitions that are within vpad km/s of either edge of
      the wavelength array.

    Returns
    -------
    tau : array of floats
      Array of optical depth values.
    """
    z = zp1 - 1
    if debug:
        i = int(len(wa)/2)
        psize =  c_kms * (wa[i] - wa[i-1]) / wa[i]
        print 'approx pixel width %.1f km/s at %.1f Ang' % (psize, wa[i])

    #select only ions with redshifted central wavelengths inside wa,
    #+/- the padding velocity range vpad.
    obswavs = ion.wa * zp1
    wmin = wa[0] * (1 - vpad / c_kms)
    wmax = wa[-1] * (1 + vpad / c_kms)
    trans = ion[between(obswavs, wmin, wmax)]
    if debug:
        if len(trans) == 0:
            print 'No transitions found overlapping with wavelength array'

    tickmarks = []
    sumtau = np.zeros_like(wa)
    i0 = i1 = None 
    for i,(wav0,osc,gam) in enumerate(trans):
        refwav = wav0 * zp1
        dv = (wa - refwav) / refwav * c_kms
        if maxdv is not None:
            i0,i1 = dv.searchsorted([-maxdv, maxdv])
        tau = calctau(dv[i0:i1], wav0, osc, gam, logN, btemp=b,
                      debug=debug, verbose=verbose)
        if ticks and tau.max() > label_tau_threshold:
            tickmarks.append((refwav, z, wav0, i))
        sumtau[i0:i1] += tau

    if ticks:
        return sumtau, tickmarks
    else:
        return sumtau

def find_tau(wa, lines, atom, per_trans=False):
    """ Given a wavelength array, a reference atom.dat file read with
    readatom, and a list of lines giving the ion, redshift,
    log10(column density) and b parameter, return the tau at each
    wavelength from all these transitions.

    Note this assumes the wavelength array has small enough pixel
    separations so that the profiles are properly sampled.
    """
    tau = np.zeros_like(wa)
    #print 'finding tau...'
    ticks = []
    ions = []
    taus = []
    for ion,z,b,logN in lines:
        #print 'z, logN, b', z, logN, b
        maxdv = 20000 if logN > 18 else 1000
        t,tick = calc_iontau(wa, atom[ion], z+1, logN, b, ticks=True,maxdv=maxdv)
        tau += t
        if per_trans:
            taus.append(t)
        ticks.extend(tick)
        ions.extend([ion]*len(tick))

    ticks = np.rec.fromarrays([ions] + zip(*ticks),names='name,wa,z,wa0,ind')

    if per_trans:
        return tau, ticks, taus
    else:
        return tau, ticks


def calc_Wr(i0, i1, wa, ew, ewer, tr):
    """ Find the rest equivalent width of a feature, and column
    density assuming optically thin.

    Parameters
    ----------
    i0, i1 : int
      Start and end indices of feature (inclusive).
    wa : array of floats, shape (N,)
      Observed wavelengths.
    ew : array of floats, shape (N,)
      Equivalent width per pixel.
    ewer : array of floats, shape (N,)
      Equivalent width 1 sigma error per pixel.
    tr : atom.dat entry
      Transition entry from an atom.dat array read by pyvpfit.readatom(),
      with attributes wav (rest wavelength) and osc (oscillator strength).

    Returns
    -------
    A dictionary with keys:

    ======== =========================================================
    logN     1 sigma low val, value, 1 sigma upper val
    Ndetlim  log N 5 sigma upper limit
    Wr       Rest equivalent width in same units as wa
    Wre      1 sigma error on rest equivalent width
    zp1      1 + redshift
    ngoodpix number of good pixels contributing to the measurements 
    Nmult    multiplier to get from equivalent width to column density
    ======== =========================================================

    """
    ew1 = np.array(ew[i0:i1+1])
    ewer1 = np.array(ewer[i0:i1+1])
    wa1 = wa[i0:i1+1]
    # interpolate over bad values
    good = ~np.isnan(ew1) & (ewer1 > 0)
    if not good.all():
        ew1[~good] = np.interp(wa1[~good], wa1[good], ew1[good])
        ewer1[~good] = np.interp(wa1[~good], wa1[good], ewer1[good]) 
    W = ew1.sum()
    We = sqrt((ewer1**2).sum())
    zp1 = 0.5*(wa1[0] + wa1[-1]) / tr['wa']
    Wr = W / zp1
    Wre = We / zp1
    # Assume we are on the linear part of curve of growth (will be an
    # underestimate if saturated)
    Nmult = 1.13e20 / (tr['osc'] * tr['wa']**2)
    chi = np.log10( Nmult * (Wr + Wre) )
    c = np.log10(Nmult * Wr)
    clo = np.log10( Nmult * (Wr - Wre) )
    Nwa = Nmult * ew1 / zp1
    # 5 sigma detection limit
    detlim = np.log10( Nmult * 5*Wre )
    return adict(logN=(clo,c,chi), Ndetlim=detlim, Wr=Wr, Wre=Wre, zp1=zp1,
                 ngoodpix=good.sum(), Nmult=Nmult)

def b_to_T(atom, bvals):
    """ Convert b parameters (km/s) to a temperature in K for an atom
    with mass amu.
    """
    amu = Ar[atom]
    b = np.atleast_1d(bvals) * 1e5

    # convert everything to cgs
    T = 0.5 * b**2 * amu * mp / k
    
    # b \propto sqrt(2kT/m)
    if len(T) == 1:
        return T[0]
    return  T

def read_HITRAN(thelot=False):
    """ Return a list of molecular absorption features in the HITRAN
    2004 list with wavelengths < 25000 Ang (Journal of Quantitative
    Spectroscopy & Radiative Transfer 96, 2005, 139-204).

    By default only lines with intensity > 5e-26 are returned. Set
    thelot=True if you really want the whole catalogue.

    The returned wavelengths are in Angstroms.

    The strongest absorption features in the optical range is
    typically O2.
    """
    filename = DATAPATH + '/linelists/HITRAN2004_wa_lt_25000.fits.gz'
    lines = readtabfits(filename)
    if not thelot:
        lines = lines[lines.intensity > 5e-26]
    lines.sort(order='wav')
    return lines

def readatom(filename=None, debug=False,
             flat=False, molecules=False, isotopes=False):
    """ Reads atomic transition data from a vpfit-style atom.dat file.

    Parameters
    ----------
    filename : str, optional
      The name of the atom.dat-style file. If not given, then the
      version bundled with `barak` is used.
    flat : bool (False)
      If True, return a flattened array, with the data not grouped by
      transition.
    molecules : bool (False)
      If True, also return data for H2 and CO molecules.
    isotopes : bool (False)
      If True, also return data for isotopes.
      
    Returns
    -------
    atom [, atom_flat] : dict [, dict]
      A dictionary of transition data, in general grouped by
      electronic transition (MgI, MgII and so on). If `flat` = True,
      also return a flattened version of the same data.
    """

    # first 2 chars - element.
    #        Check that only alphabetic characters
    #        are used (if not, discard line).
    # next 4 chars - ionization state (I, II, II*, etc)
    # remove first 6 characters, then:
    # first string - wavelength
    # second string - osc strength
    # third string - lifetime? (intrinsic width constant)
    # ignore anything else on the line

    if filename is None:
        filename = DATAPATH + '/linelists/atom.dat.gz'

    if filename.endswith('.gz'):
        import gzip
        fh = gzip.open(filename)
    else:
        fh = open(filename)

    atom = dict()
    atomflat = []

    for line in fh:
        if debug:  print line
        if not line[0].isupper():  continue
        ion = line[:6].replace(' ','')
        if not molecules:
            if ion[:2] in set(['HD','CO','H2']):
                continue
        if not isotopes:
            if ion[-1] in 'abc' or ion[:3] == 'C3I':
                continue
        wav,osc,gam = [float(item) for item in line[6:].split()[:3]]
        if ion in atom:
            atom[ion].append((wav,osc,gam))
        else:
            atom[ion] = [(wav,osc,gam)]
        atomflat.append( (ion,wav,osc,gam) )

    fh.close()
    # turn each ion into a record array
    for ion in atom:
        atom[ion] = np.rec.fromrecords(atom[ion], names='wa,osc,gam')

    atomflat = np.rec.fromrecords(atomflat,names='name,wa,osc,gam')

    if flat:
        return atom, atomflat
    else: 
        return atom

def findtrans(name, atomdat):
    """ Given an ion and wavelength and list of transitions read with
    readatom(), return the best matching entry in atom.dat.

    >>> name, tr = findtrans('CIV 1550', atomdat)
    """
    i = 0
    name = name.strip()
    if name[:4] in ['H2J0','H2J1','H2J2','H2J3','H2J4','H2J5']:
        i = 4
    else:
        while name[i].isalpha() or name[i] == '*': i += 1
    ion = name[:i]
    wa = float(name[i:])
    # must be sorted lowest to highest for indexnear
    isort = np.argsort(atomdat[ion].wa)
    sortwa = atomdat[ion].wa[isort]
    ind = indexnear(sortwa, wa)
    tr = atomdat[ion][isort[ind]]
    # Make a short string that describes the transition, like 'CIV 1550'
    wavstr = ('%.1f' % tr['wa']).split('.')[0]
    trstr =  '%s %s' % (ion, wavstr)
    return trstr, atomdat[ion][isort[ind]]

def split_trans_name(name):
    """ Given a transition string (say MgII), return the name of the
    atom and the ionization string (Mg, II).
    """
    i = 0
    while name[i] not in 'XVI':
        i += 1
    return name[:i], name[i:]
