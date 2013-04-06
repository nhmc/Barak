""" This module has routines for analysing the absorption profiles
from ions and molecules.
"""
# p2.6+ compatibility
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
if sys.version > '3':
    from io import StringIO
    basestring = str
else:
    from cStringIO import StringIO

from .voigt import voigt
from .convolve import convolve_psf
from .utilities import between, adict, get_data_path, indexnear
from .constants import Ar, me, mp, kboltz, c, e, sqrt_ln2, c_kms
from .spec import find_bin_edges
from .sed import  make_constant_dv_wa_scale
from .abundances import Asolar
from .pyvpfit import readf26

import numpy as np

import math
from math import pi, sqrt, exp


DATAPATH = get_data_path()

# this constant gets used in several functions (units of cm^2/s)
e2_me_c = e**2 / (me*c)

def calc_sigma_on_f(vel, wa0, gam, b, debug=False, verbose=True):
    """ Calculate the quantity sigma / oscillator strength.

    Multiply this by the oscillator strength and the column density
    (cm^-2) to get the optical depth.

    Parameters
    ----------
    vel : array of floats, shape (N,)
      Velocities in km/s.
    wa0 : float
      Rest wavelength of transition in Angstroms.
    gam : float
      Gamma parameter for the transition (dimensionless).
    b : float
      *b* parameter (km/s).

    Returns
    -------
    sigma_on_f: array, shape (N,)
      sigma/oscillator strength in units of cm^2.
    """
    # note units are cgs
    wa0 = wa0 * 1e-8                    # cm
    b = b * 1e5                           # cm/s
    nu0 = c / wa0                        # rest frequency, s^-1
    # Now use doppler relation between v and nu assuming gam << nu0
    gam_v = gam / nu0 * c             # cm/s
    if debug:
        print('Widths in km/s (Lorentzian Gamma, Gaussian b):',
              gam_v/1.e5, b/1.e5)

    if verbose:
        fwhml = gam_v / (2.*pi)                # cm/s
        fwhmg = 2. * sqrt_ln2 * b                # cm/s

        ##### sampling check, first get velocity width ######
        ic = np.searchsorted(vel, 0)
        try:
            # it's ok if the transition is outside the fitting region.
            if ic == len(vel):
                ic -= 1
            if ic == 0:
                ic += 1
            vstep = vel[ic] - vel[ic-1]
        except IndexError:
            raise IndexError(4*'%s ' % (len(vel), ic, ic-1, vel))

        fwhm = max(gam_v/1.e5, fwhmg/1.e5)
        
        if vstep > fwhm:
            print('Warning: tau profile undersampled!')
            print('  Pixel width: %f km/s, transition fwhm: %f km/s' % (
                vstep, fwhm))
            # best not to correct for this here, because even if we do,
            # we'll get nonsense if we convolve the resulting flux with an
            # instrumental profile.  Need to use smaller dv size
            # throughout tau, exp(-tau) and convolution of exp(-tau)
            # calculations, only re-binning back to original dv size after
            # all these steps.

    u = 1.e5 / b * vel                         # dimensionless
    a = gam_v / (4*pi*b)                       # dimensionless
    vp = voigt(a, u)                           # dimensionless
    
    # Note the below isn't exactly the same as sigma as defined by
    # Draine et al. It must be multiplied by the oscillator strength
    # and the column density to give the optical depth.
    sigma = pi * e2_me_c * wa0 / (sqrt(pi) * b) * vp

    return sigma


def calctau(vel, wa0, osc, gam, logN, b, debug=False, verbose=True):
    """ Returns the optical depth (Voigt profile) for a transition.

    Given an transition with rest wavelength wa0, osc strength,
    natural linewidth gam; b parameter (doppler and turbulent); and
    log10 (column density), returns the optical depth in velocity
    space. v is an array of velocity values in km/s. The resulting
    absorption feature is centred at v=0.

    Parameters
    ----------
    vel : array of floats, shape (N,)
      Velocities in km/s.
    wa0 : float
      Rest wavelength of transition in Angstroms.
    osc : float
      Oscillator strength of transition (dimensionless).
    gam : float
      Gamma parameter for the transition (dimensionless).
    logN : float:
      log10 of the column density in absorbers per cm^2.
    b : float
      *b* parameter (km/s).

    Returns
    -------
    tau : array of floats, shape (N,)
      The optical depth as a function of `vel`.

    Notes
    -----
    The step size for `vel` must be small enough to properly
    sample the profile.

    To map the velocity array to some wavelength for a transition
    with rest wavelength `wa0` at redshift `z`:

    >>> z = 3.0
    >>> wa = wa0 * (1 + z) * (1 + v/c_kms)
    """
    temp = calc_sigma_on_f(vel, wa0, gam, b, debug=debug, verbose=verbose)
    # note units are cgs
    return (10**logN * osc) * temp


def calc_tau_peak(logN, b, wa0, osc):
    """ Find the optical depth of a transition at line centre assuming
    we are on the linear part of the curve of growth.

    Parameters
    ----------
    logN : array_like
      log10 of column density in cm^-2
    b : float
      b parameter in km/s.
    wa0 : float
      Rest wavelength of the transition in Angstroms.
    osc : float
      Transition oscillator strength.

    Returns
    -------
    tau0 : ndarray or scalar if logN is scalar
      optical depth at the centre of the line

    Notes
    -----
    See Draine "Physics of the Interstellar and Intergalactic medium"
    for a description of how to calculate this quantity.
    """
    b_cm_s = b * 1e5

    wa0 = wa0 * 1e-8 # cm
    return sqrt(pi) * e2_me_c * 10**logN * osc * wa0 / b_cm_s

def logN_from_tau_peak(tau, b, wa0, osc):
    """ Calculate the column density for a transition given its
    width and its optical depth at line centre.
    
    Parameters
    ----------
    tau : array_like
      Optical depth at line centre.
    b : float
      b parameter in km/s.
    wa0 : float
      Rest wavelength of the transition in Angstroms.
    osc : float
      Transition oscillator strength.

    Returns
    -------
    logN : ndarray, or scalar if tau is scalar
      log10 of column density in cm^-2

    See Also
    --------
    calc_tau_peak
    """
    b_cm_s = b * 1e5

    wa0 = wa0 * 1e-8 # cm

    return np.log10(tau * b_cm_s / (sqrt(pi) * e2_me_c * osc * wa0))


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
        print('approx pixel width %.1f km/s at %.1f Ang' % (psize, wa[i]))

    #select only ions with redshifted central wavelengths inside wa,
    #+/- the padding velocity range vpad.
    obswavs = ion.wa * zp1
    wmin = wa[0] * (1 - vpad / c_kms)
    wmax = wa[-1] * (1 + vpad / c_kms)
    trans = ion[between(obswavs, wmin, wmax)]
    if debug:
        if len(trans) == 0:
            print('No transitions found overlapping with wavelength array')

    tickmarks = []
    sumtau = np.zeros_like(wa)
    i0 = i1 = None 
    for i,(wa0,osc,gam) in enumerate(trans):
        tau0 = calc_tau_peak(logN, b, wa0, osc)
        if 1 - exp(-tau0) < 1e-3:
            continue
        refwav = wa0 * zp1
        dv = (wa - refwav) / refwav * c_kms
        if maxdv is not None:
            i0,i1 = dv.searchsorted([-maxdv, maxdv])
        tau = calctau(dv[i0:i1], wa0, osc, gam, logN, b,
                      debug=debug, verbose=verbose)
        if ticks and tau0 > label_tau_threshold:
            tickmarks.append((refwav, z, wa0, i))
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

    lines can also be the name of a VPFIT fort.26 format file or a
    VpfitModel.lines record array.

    Note this assumes the wavelength array has small enough pixel
    separations so that the profiles are properly sampled.
    """
    try:
        vp = readf26(lines)
    except AttributeError:
        pass
    if hasattr(lines, 'dtype'):
        lines = [(l['name'].replace(' ', ''), l['z'], l['b'], l['logN'])
                 for l in lines]
        
    tau = np.zeros_like(wa)
    #print 'finding tau...'
    ticks = []
    ions = []
    taus = []
    for ion,z,b,logN in lines:
        #print 'z, logN, b', z, logN, b
        maxdv = 20000 if logN > 18 else 1000
        t,tick = calc_iontau(wa, atom[ion], z+1, logN, b, ticks=True,
                             maxdv=maxdv)
        tau += t
        if per_trans:
            taus.append(t)
        ticks.extend(tick)
        ions.extend([ion]*len(tick))

    ticks = np.rec.fromarrays([ions] + list(zip(*ticks)),
                              names=str('name,wa,z,wa0,ind'))

    if per_trans:
        return tau, ticks, taus
    else:
        return tau, ticks


def calc_Wr(i0, i1, wa, tr, ew=None, ewer=None, fl=None, er=None, co=None,
            cohi=0.05, colo=0.05):
    """ Find the rest equivalent width of a feature, and column
    density assuming optically thin.

    You must give either fl, er and co, or ew and ewer.

    Parameters
    ----------
    i0, i1 : int
      Start and end indices of feature (inclusive).
    wa : array of floats, shape (N,)
      Observed wavelengths.
    tr : atom.dat entry
      Transition entry from an atom.dat array read by `readatom()`.
    ew : array of floats, shape (N,), optional
      Equivalent width per pixel.
    ewer : array of floats, shape (N,), optional
      Equivalent width 1 sigma error per pixel.
      with attributes wav (rest wavelength) and osc (oscillator strength).
    fl : array of floats, shape (N,), optional
      Observed flux.
    er : array of floats, shape (N,), optional
      Observed flux 1 sigma error.
    co : array of floats, shape (N,), optional
      Observed continuum.
    cohi : float (0.05)
      When calculating one sigma upper error and detection limit,
      increase the continuum by this fractional amount. Only used if
      fl, er and co are also given.
    colo : float (0.05)
      When calculating one sigma lower error decrease the continuum by
      this fractional amount.  Only used if fl, er and co are also
      given.
      
    Returns
    -------
    A dictionary with keys:

    ========= =========================================================
    logN      1 sigma low val, value, 1 sigma upper val
    Ndetlim   log N 5 sigma upper limit
    Wr        Rest equivalent width in same units as wa
    Wre       1 sigma error on rest equivalent width
    zp1       1 + redshift
    ngoodpix  number of good pixels contributing to the measurements 
    Nmult     multiplier to get from equivalent width to column density
    saturated Are more than 10% of the pixels between limits saturated?
    ========= =========================================================
    """
    wa1 = wa[i0:i1+1]

    npts = i1 - i0 + 1
    # if at least this many points are saturated, then mark the
    # transition as saturated
    n_saturated_thresh = int(0.1 * npts)
    saturated = None
    
    if ew is None:
        assert None not in (fl, er, co)
        wedge = find_bin_edges(wa1)
        dw = wedge[1:] - wedge[:-1]
        ew1 = dw * (1 - fl[i0:i1+1] / co[i0:i1+1]) 
        ewer1 = dw * er[i0:i1+1] / co[i0:i1+1]
        c = (1 + cohi) * co[i0:i1+1]
        ew1hi = dw * (1 - fl[i0:i1+1] / c)
        ewer1hi = dw * er[i0:i1+1] / c
        c = (1 - colo) * co[i0:i1+1]
        ew1lo = dw * (1 - fl[i0:i1+1] / c)
        ewer1lo = dw * er[i0:i1+1] / c
        if (fl[i0:i1+1] < er[i0:i1+1]).sum() >= n_saturated_thresh:
            saturated = True
        else:
            saturated = False
        
    else:
        assert None not in (ew, ewer)
        ew1 = np.array(ew[i0:i1+1])
        ewer1 = np.array(ewer[i0:i1+1])

    # interpolate over bad values
    good = ~np.isnan(ew1) & (ewer1 > 0)
    if not good.any():
        return None
    if not good.all():
        ew1[~good] = np.interp(wa1[~good], wa1[good], ew1[good])
        ewer1[~good] = np.interp(wa1[~good], wa1[good], ewer1[good]) 
        if fl is not None:
            ew1hi[~good] = np.interp(wa1[~good], wa1[good], ew1hi[good])
            ewer1hi[~good] = np.interp(wa1[~good], wa1[good], ewer1hi[good]) 
            ew1lo[~good] = np.interp(wa1[~good], wa1[good], ew1lo[good])
            ewer1lo[~good] = np.interp(wa1[~good], wa1[good], ewer1lo[good]) 

    W = ew1.sum()
    We = sqrt((ewer1**2).sum())
    if fl is not None:
        Whi = ew1hi.sum()
        Wehi = sqrt((ewer1hi**2).sum())
        Wlo = ew1lo.sum()
        Welo = sqrt((ewer1lo**2).sum())

    zp1 = 0.5*(wa1[0] + wa1[-1]) / tr['wa']
    Wr = W / zp1
    Wre = We / zp1
    if fl is not None:
        Wrhi = Whi / zp1
        Wrehi = Wehi / zp1
        Wrlo = Wlo / zp1
        Wrelo = Welo / zp1
    
    # Assume we are on the linear part of curve of growth (will be an
    # underestimate if saturated). See Draine, "Physics of the
    # Interstellar and Intergalactic medium", ISBN 978-0-691-12214-4,
    # chapter 9.
    Nmult = 1.13e20 / (tr['osc'] * tr['wa']**2)

    # 5 sigma detection limit
    detlim = np.log10( Nmult * 5*Wre )

    logN = np.log10(Nmult * Wr)
    if fl is not None:
        logNhi = np.log10( Nmult * (Wrhi + Wrehi) )
        logNlo = np.log10( Nmult * (Wrlo - Wrelo) )
    else:
        logNhi = np.log10( Nmult * (Wr + Wre) )
        logNlo = np.log10( Nmult * (Wr - Wre) )

    return adict(logN=(logNlo,logN,logNhi), Ndetlim=detlim,
                 Wr=Wr, Wre=Wre, zp1=zp1,
                 ngoodpix=good.sum(), Nmult=Nmult, saturated=saturated)

def b_to_T(atom, bvals):
    """ Convert b parameters (km/s) to a temperature in K for an atom
    with mass amu.

    Parameters
    ----------
    atom : str or float
      Either an abbreviation for an element name (for example 'Mg'),
      or a mass in amu.
    bvals : array_like
      One or more b values in km/s

    Returns
    -------
    T : ndarray or float
      The temperature corresponding to each value in `bvals'.
    """
    if isinstance(atom, basestring):
        amu = Ar[atom]
    else:
        amu = float(atom)

    b = np.atleast_1d(bvals) * 1e5

    # convert everything to cgs
    T = 0.5 * b**2 * amu * mp / kboltz
    
    # b \propto sqrt(2kT/m)
    if len(T) == 1:
        return T[0]
    return  T

def T_to_b(atom, T):
    """ Convert temperatures in K to b parameters (km/s) for an atom
    with mass amu.

    Parameters
    ----------
    atom : str or float
      Either an abbreviation for an element name (for example 'Mg'),
      or a mass in amu.
    T : array_like
      One or more temperatures in Kelvin.

    Returns
    -------
    b : ndarray or float
      The b value in km/s corresponding to each input temperature.
    """
    if isinstance(atom, basestring):
        amu = Ar[atom]
    else:
        amu = float(atom)

    T = np.atleast_1d(T)
    b_cms = np.sqrt(2 * kboltz * T / (mp *amu))
    
    b_kms = b_cms / 1e5

    # b \propto sqrt(2kT/m)
    if len(b_kms) == 1:
        return b_kms[0]
    return  b_kms


def read_HITRAN(thelot=False):
    """ Return a list of molecular absorption features in the HITRAN
    2004 list with wavelengths < 25000 Ang (Journal of Quantitative
    Spectroscopy & Radiative Transfer 96, 2005, 139-204).

    By default only lines with intensity > 5e-26 are returned. Set
    thelot=True if you really want the whole catalogue.

    The returned wavelengths are in Angstroms.

    The strongest absorption features in the optical range are
    typically due to O2.
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
        filename = DATAPATH + '/linelists/atom.dat'

    if filename.endswith('.gz'):
        import gzip
        fh = gzip.open(filename, 'rb')
    else:
        fh = open(filename, 'rb')

    atom = dict()
    atomflat = []
    specials = set(['??', '__', '>>', '<<', '<>'])
    for line in fh:
        line = line.decode('utf-8')
        if debug:  print(line)
        if not line[0].isupper() and line[:2] not in specials:
            continue
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
        atom[ion] = np.rec.fromrecords(atom[ion], names=str('wa,osc,gam'))

    atomflat = np.rec.fromrecords(atomflat,names=str('name,wa,osc,gam'))

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
    i = 1
    while name[i] not in 'XVI':
        i += 1
    return name[:i], name[i:]

def tau_LL(logN, wa, wstart=912):
    """ Find the optical depth at the neutral hydrogen Lyman limit.

    Parameters
    ----------
    logN : float
      log10 of neutral hydrogen column density in cm^-2.
    wa : array_like
      Wavelength in Angstroms.
    wstart : float (912.)
      Tau values at wavelengths above this are set to zero.

    Returns
    -------
    tau : ndarray or float if `wa` is scalar
      The optical depth at each wavelength.

    Notes
    -----
    At the Lyman limit, the optical depth tau is given by::

      tau = N(HI) * sigma_0

    where sigma_0 = 6.304 e-18 cm^2 and N(HI) is the HI column density
    in cm^-2. The energy dependence of the cross section is::

      sigma_nu ~ sigma_0 * (h*nu / I_H)^-3 = sigma_0 * (lam / 912)^3

    where I_H is the energy needed to ionise hydrogen (1 Rydberg, 13.6
    eV), nu is frequency and lam is the wavelength in Angstroms. This
    expression is valid for I_H < I < 100* I_H.

    So the normalised continuum bluewards of the Lyman limit is::

      F/F_cont = exp(-tau) = exp(-N(HI) * sigma_lam)
               = exp(-N(HI) * sigma_0 * (lam/912)^3)

    Where F is the absorbed flux and F_cont is the unabsorbed
    continuum.

    References
    ----------
    Draine, 2011, "Physics of the Interstellar Medium".
    ISBN 978-0-691-12214-4: p84, and p128 for the photoionization cross
    section.

    Examples
    --------
    >>> wa = np.linspace(100, 912, 100)
    >>> z = 2.24
    >>> for logN in np.arange(17, 21., 0.5):
    ...    fl = exp(-tau_LL(logN, wa))
    ...    plt.plot(wa*(1+z), fl, lw=2, label='%.2f' % logN)
    >>> plt.legend()
    """
    sigma0 = 6.304e-18           # cm^2
    i = wa.searchsorted(wstart)
    tau = np.zeros_like(wa)
    tau[:i] = 10**logN * sigma0 * (wa[:i] / 912.)**3 
    return tau

def calc_DLA_tau(wa, z=0, logN=20.3, logZ=0, bHI=50, atom=None,
                 verbose=1, highions=1, molecules=False):
    """ Create the optical depth due to absorption from a DLA.

    The DLA is at z=0. The column density and metallicity can be
    varied. Solar Abundance ratios are assumed, with most of the atoms
    in the singly ionised state

    Parameters
    ----------
    wa : array_like
       Wavelength scale.
    logZ : float (0)
       log10 of the metal abundance relative to solar. For example 0
       (the default) gives solar abundances, -1 gives 1/10th of solar.
    verbose : bool (False)
       Print helpful information

    Returns
    -------
    tau, ticks : ndarrays, structured array
      tau at each wavelength and tick positions and names.
    """
    off = -12 + logN + logZ
    temp = b_to_T('H', bHI)
    print('b %.2f km/s gives a temperature of %.1f K' % (bHI, temp))
    
    elements = 'O Si Fe C Ca Al Ti N Zn Cr'.split()
    b = dict((el, T_to_b(el, temp)) for el in elements)
    print('using low ion b values:')
    print(', '.join('%s %.1f' % (el, b[el]) for el in elements))
    
    f26 = [
        'HI    %.6f  0  %.2f 0 %.2f 0' % (z, bHI, logN),    
        'OI    %.6f  0  %.2f 0 %.2f 0' % (z, b['O'], (Asolar['O']  + off)), 
        'SiII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Si'], (Asolar['Si'] + off - 0.05)), 
        'SiIII %.6f  0  %.2f 0 %.2f 0' % (z, b['Si'], (Asolar['Si'] + off - 1)),    
        'FeII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Fe'], (Asolar['Fe'] + off)),    
        'CII   %.6f  0  %.2f 0 %.2f 0' % (z, b['C'], (Asolar['C']  + off - 0.05)),    
        'CIII  %.6f  0  %.2f 0 %.2f 0' % (z, b['C'], (Asolar['C']  + off - 1)),    
        'CaII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Ca'], (Asolar['Ca'] + off)),
        'AlII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Al'], (Asolar['Al'] + off - 0.05)),
        'AlIII %.6f  0  %.2f 0 %.2f 0' % (z, b['Al'], (Asolar['Al'] + off - 1)),
        'TiII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Ti'], (Asolar['Ti'] + off)),
        'NII   %.6f  0  %.2f 0 %.2f 0' % (z, b['N'], (Asolar['N']  + off)),
        'ZnII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Zn'], (Asolar['Zn']  + off)),
        'CrII  %.6f  0  %.2f 0 %.2f 0' % (z, b['Cr'], (Asolar['Cr']  + off)),
        ]
    if highions:
        print('Including O VI, Si IV, C IV and N V')
        logNCIV = 15.
        logNSiIV = logNCIV - (Asolar['C'] - Asolar['Si'] )
        f26 = f26 + [
            'OVI   %.6f  0  30.0 0 15.0 0' % z,
            'SiIV  %.6f  0  %.2f  0 %.2f 0' % (z, b['Si'], logNSiIV),
            'CIV   %.6f  0  %.2f 0 %.2f 0' % (z, b['C'], logNCIV),
            'NV    %.6f  0  30.0 0 15.0 0' % z,
            ]
    if molecules:
        print('Including H_2 and CO')
        f26 = f26 + [
            'H2J0  %.6f  0  5.0  0 18.0 0' % z,
            'H2J1  %.6f  0  5.0  0 19.0 0' % z,
            'COJ0  %.6f  0  5.0  0 14.0 0' % z,
            ]

    f26 = StringIO('\n'.join(f26))
    if atom is None:
        atom = readatom(molecules=molecules)

    tau,ticks = find_tau(wa, f26, atom)
    tau += tau_LL(logN, wa/(1+z), wstart=912.5)
    return tau, ticks

def calc_DLA_trans(wa, redshift, vfwhm, logN=20.3, logZ=0, bHI=50,
                   highions=True, molecules=False):
    """ Find the transmission after absorption by a DLA

    Parameters
    ----------
    wa : array_like, shape (N,)
      wavelength array.
    redshift : float
      The redshift of the DLA.
    vfwhm : float
      The resolution FWHM in km/s.
    logZ : float (0)
       log10 of the metal abundance relative to solar. For example 0
       (the default) gives solar abundances, -1 gives 1/10th of solar.
    bHI : float (40)
       b parameter for the HI. Other species are assumed to be
       thermally broadened.

    Returns
    -------
    transmission : ndarray, shape (N,)
      The transmission at each wavelength
    ticks : ndarray, shape (M,)
      Information for making ticks to show absorbing components.
    """
    wa1 = make_constant_dv_wa_scale(wa[0], wa[-1], vfwhm/3.)  
    tau, ticks = calc_DLA_tau(
        wa1, z=redshift, logN=logN, logZ=logZ, bHI=bHI,
        highions=highions, molecules=molecules)
    trans = convolve_psf(np.exp(-tau),  3.)
    trans1 = np.interp(wa, wa1, trans)
    return trans1, ticks


def guess_logN_b(ion, wa0, osc, tau0):
    """ Estimate logN and b for a transition given the peak optical
    depth and atom.

    Examples
    --------
    >>> logN, b = guess_b_logN('HI', 1215.6701, nfl, ner)
    """

    T = 10000
    if ion == 'HI':
        T = 30000
    elif ion in ('CIV', 'SiIV'):
        T = 20000
        
    b = T_to_b(split_trans_name(ion)[0], T)

    if ion == 'OVI':
        b = 30.

    return logN_from_tau_peak(tau0, b, wa0, osc), b

def Nvel_from_tau(tau, wa, osc):
    """ Returns the N_vel, the column density per velocity interval.

    Parameters
    ----------
    tau : array_like, shape (N,)
      Array of optical depths/    
    wa : float
      Rest wavelength of the transition.
    osc : float
      Transition oscillator strength.

    Returns
    -------
    N_vel : ndarray, shape (N,)
      The column density per velocity interval, with units cm^-2
      (km/s)^-1 Multiply this by a velocity interval in km/s to get a
      column density.
    """
    # the 1e5 here converts from (cm/s)^-1 to (km/s)^-1 
    return tau / (pi * e2_me_c * osc) / (wa * 1e-8) * 1e5


def Nlam_from_tau(tau, wa, osc):
    """ Returns the N_lambda, the column density per rest wavelength
    interval.

    Parameters
    ----------
    tau : array_like, shape (N,)
      Array of optical depths/    
    wa : float
      Rest wavelength of the transition.
    osc : float
      Transition oscillator strength.

    Returns
    -------
    Nlam : ndarray, shape (N,)
      The column density per wavelength interval, with units cm^-2
      Angstrom^-1. Multiply this by a rest wavelength interval in
      Angstroms to get a column density.
    """
    # the 1e-8 here converts from cm^-1 to Angstrom^-1 
    return tau / (pi * e2_me_c * osc) * c / (wa * 1e-8)**2 * 1e-8

def log10N_from_Wr(Wr, wa0, osc):
    """ Find log10(Column density) from a rest frame equivalent width
    assuming optically thin.

    Parameters
    ----------
    Wr : float
       Rest frame equivalent width in Angstroms.
    wa0 : float
       Transition wavelength in Angstroms.
    osc: float
       Transition oscillator strength.

    Returns
    -------
    log10N : float
      log10(column density in cm^-2), or zero if the equivalent width
      is 0 or negative.

    Notes
    -----
    Assumes we are on the linear part of curve of growth (will be an
    underestimate if saturated). See Draine,"Physics of the
    Interstellar and Intergalactic medium", ISBN 978-0-691-12214-4,
    chapter 9.
    """
    if not Wr > 0:
        return 0

    Nmult = 1.13e20 / (osc * wa0**2)    
    return np.log10(Nmult * Wr)
