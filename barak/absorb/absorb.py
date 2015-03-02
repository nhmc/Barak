""" This module has routines for analysing the absorption profiles
from ions and molecules.
"""
# p2.6+ compatibility
from __future__ import division, print_function, unicode_literals

import sys
if sys.version > '3':
    from io import StringIO
    basestring = str
else:
    from cStringIO import StringIO

from .voigt import voigt
from ..convolve import convolve_psf
from ..utilities import between, get_data_path, indexnear
from ..constants import Ar, sqrt_ln2
from ..sed import  make_constant_dv_wa_scale, vel_from_wa
from ..abundances import Asolar
from ..pyvpfit import readf26
from ..io import readatom

import numpy as np

import astropy.constants as C
import astropy.units as u

import math
from math import pi, sqrt, exp

__all__ = ['calc_iontau', 'find_tau' , 'b_to_T', 'T_to_b', 'findtrans',
           'split_trans_name', 'tau_LL', 'calc_DLA_tau', 'calc_DLA_trans',
           'guess_logN_b', 'get_ionization_energy',
           'photo_cross_section_hydrogenic', 'readatom']


DATAPATH = get_data_path()

ION_CACHE = {}
ATOMDAT = None


kboltz = C.k_B.to(u.erg / u.K)
mp_g = C.m_p.to(u.g).value
c_kms = C.c.to(u.km / u.s).value
c_cms = C.c.to(u.cm / u.s).value
# this constant gets used in several functions
e2_me_c = (C.e.gauss**2 / (C.m_e*C.c)).to(u.cm**2 / u.s).value

def _get_atomdat():
    """ Function to cache atom.dat"""
    global ATOMDAT
    if ATOMDAT is None:
        ATOMDAT = readatom(molecules=True)

    return ATOMDAT

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
    nu0 = c_cms / wa0                        # rest frequency, s^-1
    # Now use doppler relation between v and nu assuming gam << nu0
    gam_v = gam / nu0 * c_cms             # cm/s
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
    See Draine "Physics of the Interstellar and Intergalactic medium".
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
                label_tau_threshold=0.01, vpad=500., verbose=True,
                logNthresh_LL=None):
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
      b parameter (km/s).
    maxdv : float (default 1000)
      For performance reasons, only calculate the Voigt profile for a
      single line to +/- maxdv.  Increase this if you expect DLA-type
      extended wings. None for no maximum.
    vpad : float (default 500)
      Include transitions that are within vpad km/s of either edge of
      the wavelength array.
    logNthresh_LL : float (default 14.8)
      Threshold value of log10(NHI) for including Lyman limit absorption.
    Returns
    -------
    tau : array of floats
      Array of optical depth values.

    or

    tau, tickmarks : arrays of floats and record array
      Optical depths and tick mark info.
    """
    if logNthresh_LL is None:
        logNthresh_LL = 14.8

    z = zp1 - 1
    if debug:
        i = int(len(wa)/2)
        psize =  c_kms * (wa[i] - wa[i-1]) / wa[i]
        print('approx pixel width %.1f km/s at %.1f Ang' % (psize, wa[i]))

    # select only ions with redshifted central wavelengths inside wa,
    # +/- the padding velocity range vpad.
    obswavs = ion.wa * zp1
    wmin = wa[0] * (1 - vpad / c_kms)
    wmax = wa[-1] * (1 + vpad / c_kms)
    trans = ion[between(obswavs, wmin, wmax)]
    if debug and len(trans) == 0:
        print('No transitions found overlapping with wavelength array')

    tickmarks = []
    sumtau = np.zeros_like(wa)
    i0 = i1 = None
    for i, (wa0, osc, gam) in enumerate(trans):
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

    if logN > logNthresh_LL and abs(ion['wa'][0] - 1215.6701) < 1e-3:
        wstart_LL = 912.8
        # remove tau from lines that move into the LL approximation
        # region.
        c0 = wa < (wstart_LL * (1+z))
        sumtau[c0] = 0
        sumtau += tau_LL(logN, wa/(1+z), wstart=wstart_LL)

    if ticks:
        return sumtau, tickmarks
    else:
        return sumtau

def find_tau(wa, lines, atomdat=None, per_trans=False, debug=False,
             logNthresh_LL=None):
    """ Given a wavelength array, a reference atom.dat file read with
    readatom, and a list of lines giving the ion, redshift,
    log10(column density) and b parameter, return the tau at each
    wavelength from all these transitions.

    lines can also be the name of a VPFIT fort.26 format file or a
    VpfitModel.lines record array.

    Note this assumes the wavelength array has small enough pixel
    separations so that the profiles are properly sampled.
    """
    if atomdat is None:
        atomdat = _get_atomdat()

    try:
        vp = readf26(lines)
    except AttributeError:
        pass
    else:
        if debug:
            print('Lines read from %s' % lines)
        lines = vp.lines

    if hasattr(lines, 'dtype'):
        if debug:
            print('Looks like a VpfitModel.lines-style record array.')
        lines = [(l['name'].replace(' ', ''), l['z'], l['b'], l['logN'])
                 for l in lines]

    tau = np.zeros_like(wa)
    if debug:
        print('finding tau...')
    ticks = []
    ions = []
    taus = []
    for ion,z,b,logN in lines:
        if debug:
            print('z, logN, b', z, logN, b)
        maxdv = 20000 if logN > 18 else 1000
        t,tick = calc_iontau(wa, atomdat[ion], z+1, logN, b, ticks=True,
                             maxdv=maxdv, logNthresh_LL=logNthresh_LL)
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


def b_to_T(atom, bvals):
    """ Convert b parameters (km/s) to a temperature in K for an atom
    with mass amu.

    Parameters
    ----------
    atom : str or float
      Either an abbreviation for an element name (for example 'Mg'),
      or a mass in amu.
    bvals : array_like
      One or more b values in km/s.

    Returns
    -------
    T : ndarray or float
      The temperature corresponding to each value in `bvals`.
    """
    if isinstance(atom, basestring):
        amu = Ar[atom]
    else:
        amu = float(atom)

    b = np.atleast_1d(bvals) * 1e5

    # convert everything to cgs
    T = 0.5 * b**2 * amu * mp_g / kboltz

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



def findtrans(name, atomdat=None):
    """ Given an ion and wavelength and list of transitions read with
    readatom(), return the best matching entry in atom.dat.

    >>> name, tr = findtrans('CIV 1550')
    """
    if atomdat is None:
        atomdat = _get_atomdat()
    i = 0
    name = name.strip()
    if name[:4] in ['H2J0','H2J1','H2J2','H2J3','H2J4','H2J5','H2J6','H2J7',
                    'COJ0','COJ1','COJ2','COJ3','COJ4','COJ5']:
        i = 4
    elif name[:3] == 'HI2':
        i = 3
    else:
        while i < len(name) and (name[i].isalpha() or name[i] == '*'):
            i += 1
    ion = name[:i]
    # must be sorted lowest to highest for indexnear
    isort = np.argsort(atomdat[ion].wa)
    sortwa = atomdat[ion].wa[isort]

    try:
        wa = float(name[i:])
    except:
        print('Possible transitions for', ion)
        print(sortwa)
        return

    ind = indexnear(sortwa, wa)
    tr = atomdat[ion][isort[ind]]
    # Make a short string that describes the transition, like 'CIV 1550'
    wavstr = ('%.1f' % tr['wa']).split('.')[0]
    trstr =  '%s %s' % (ion, wavstr)
    return trstr, atomdat[ion][isort[ind]].copy()

def split_trans_name(name):
    """ Given a transition string (say MgII), return the name of the
    atom and the ionization string (Mg, II).
    """
    i = 1
    while name[i] not in 'XVI':
        i += 1
    return name[:i], name[i:]

def photo_cross_section_hydrogenic(E, Z=1):
    """ The photoionization cross section of absorption for a
    hydrogenic (hydrogen-like, single outer electron) atom with charge
    Z.

    Parameters
    ----------
    E : array of shape (N,)
      The energy (h nu) in rydbergs.

    Z : int (default 1)
      Atomic charge. 1 for hydrogen HI, 2 for HeII, etc.

    Returns
    -------
    sigma : array shape (N,)
      Cross section in cm^2

    Examples
    --------
    Convert from, say, Ryd to Angstroms with:

    >>> import astropy.units as u
    >>> E.to(u.AA, equivalencies=u.equivalencies.spectral())

    >>> E = 10**np.linspace(1,4)
    >>> sigma = photo_cross_section_hydrogenic(E)

    >>> plt.loglog(E, sigma)
    """

    Z = float(Z)
    E = np.asarray(E)
    E0 = Z**2
    c0 = E >= E0
    out = np.zeros(len(E), float)
    sigma0 = 6.304e-18 * Z**-2
    if c0.any():
        Enorm = E[c0] / E0
        x = np.sqrt(Enorm - 1)   # unitless
        out[c0] = sigma0 * Enorm**-4 * \
                   np.exp(4 - 4*np.arctan(x)/x) / (1 - np.exp(-2*pi/x))
    return out

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
                 verbose=1, highions=1, molecules=False, same_b=False,
                 metals=True):
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
    bHI : float (50)
       The b parameter to use for HI.
    same_b : True
       Use the same b parameter for all species.
    molecules : bool (False)
       Whether to include absorption from H2 and CO.
    verbose : bool (False)
       Print helpful information
    highions : bool (True)
       Whether to include absorption from CIV, SiIV, NV and OVI

    Returns
    -------
    tau, ticks : ndarrays, structured array
      tau at each wavelength and tick positions and names.
    """
    if atom is None:
        atom = _get_atomdat()

    f26 = [
        'HI    %.6f  0  %.2f 0 %.2f 0' % (z, bHI, logN),
        ]

    if metals:
        off = -12 + logN + logZ
        temp = b_to_T('H', bHI)
        if verbose:
            print('b %.2f km/s gives a temperature of %.1f K' % (bHI, temp))

        elements = 'O Si Fe C Ca Al Ti N Zn Cr'.split()
        if same_b:
            b = dict((el, bHI) for el in elements)
        else:
            b = dict((el, T_to_b(el, temp)) for el in elements)
        if verbose:
            print('using low ion b values:')
            print(', '.join('%s %.1f' % (el, b[el]) for el in elements))

        f26 = f26 + [
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
    if highions and metals:
        if verbose:
            print('Including O VI, Si IV, C IV and N V')
        logNCIV = 14.
        logNSiIV = logNCIV - (Asolar['C'] - Asolar['Si'] )
        f26 = f26 + [
            'OVI   %.6f  0  30.0 0 14.5 0' % z,
            'SiIV  %.6f  0  %.2f  0 %.2f 0' % (z, b['Si'], logNSiIV),
            'CIV   %.6f  0  %.2f 0 %.2f 0' % (z, b['C'], logNCIV),
            'NV    %.6f  0  30.0 0 14.0 0' % z,
            ]
    if molecules:
        if verbose:
            print('Including H_2 and CO')
        f26 = f26 + [
            'H2J0  %.6f  0  5.0  0 18.0 0' % z,
            'H2J1  %.6f  0  5.0  0 19.0 0' % z,
            'COJ0  %.6f  0  5.0  0 14.0 0' % z,
            ]

    tau, ticks = find_tau(wa, StringIO('\n'.join(f26)), atomdat=atom)
    if str('wa') in ticks.dtype.names:
        ticks.sort(order=str('wa'))

    return tau, ticks

def calc_DLA_trans(wa, redshift, vfwhm, logN=20.3, logZ=0, bHI=50,
                   highions=True, molecules=False, same_b=False,
                   verbose=True, metals=False):
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
    npix = max(3, int(vfwhm/30.))
    wa1 = make_constant_dv_wa_scale(wa[0], wa[-1], vfwhm/npix)
    tau, ticks = calc_DLA_tau(
        wa1, z=redshift, logN=logN, logZ=logZ, bHI=bHI,
        highions=highions, molecules=molecules, same_b=same_b,
        verbose=verbose, metals=metals)
    trans = convolve_psf(np.exp(-tau),  npix)
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


def get_ionization_energy(species):
    """ Find the ionization energy for a species.

    Uses table 4 in Verner et al., 94.

    Parameters
    ----------
    species : str or list of str
      For example 'CII'.

    Returns
    -------
    energy : float or array of floats
      Ionization energy in eV.

    Examples
    --------
    energy = get_ionization_energy('CII')
    """
    global ION_CACHE
    if 'table' not in ION_CACHE:
        from astropy.table import Table
        ION_CACHE['table'] = Table.read(
            DATAPATH + '/ionization_energies/IE.fits')
        ION_CACHE['row_map'] = {s:i for i,s in enumerate(
            ION_CACHE['table']['name'])}

    if isinstance(species, basestring):
        i = ION_CACHE['row_map'][species]
        return ION_CACHE['table']['IE'][i]
    else:
        ind = [ION_CACHE['row_map'][s] for s in species]
        return np.array([ION_CACHE['table']['IE'][i] for i in ind],
                        dtype=float)

