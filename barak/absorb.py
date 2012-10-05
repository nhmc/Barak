""" This module has routines for analysing the absorption profiles
from ions and molecules.
"""
import voigt
from convolve import convolve_psf
from utilities import between, adict, get_data_path, indexnear
from constants import Ar, me, mp, kboltz, c, e, sqrt_ln2, c_kms
from spec import find_wa_edges, make_constant_dv_wa_scale
from abundances import Asolar
from pyvpfit import readf26

import numpy as np

from cStringIO import StringIO
import math
from math import pi, sqrt


DATAPATH = get_data_path()

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
        btemp = sqrt(2*kboltz*T / mp) * 1e-5     # km/s
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

    u = 1.e5 / b * v                           # dimensionless
    a = gam_v / (4*pi*b)                       # dimensionless
    vp = voigt.voigt(a, u)                     # dimensionless
    const = pi * e**2 / (me*c)                 # m^2/s
    tau = const*N*osc*wav0 / (sqrt(pi)*b) * vp # dimensionless

    return tau

def calc_tau_peak(logN, b, trans):
    """
    Estimate of  the peak optical depth of a transition assuming we are on the
    linear part of the curve of growth.

    logN is log10 of column density in cm^-2
    b is b parameter in km/s
    trans is  a transisition, e.e.g 'HI 1215'
    """
    b_cm_s = b * 1e5
    if isinstance(trans, basestring):
        tname, trans = findtrans(trans, readatom())
    wav0 = trans[0] * 1e-8 # cm
    osc = trans[1]
    const = pi * e**2 / (me*c)
    #gauss_norm = 1 / (sigma * sqrt(2*pi))
    return 10**logN * osc * wav0 * const / (b_cm_s * sqrt(pi))


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

    lines can also be the name of a fort.26 format file

    Note this assumes the wavelength array has small enough pixel
    separations so that the profiles are properly sampled.
    """
    try:
        vp = readf26(lines)
    except AttributeError:
        pass
    else:
        lines = [(l['name'].strip(), l['z'], l['b'], l['logN'])
                 for l in vp.lines]

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


def calc_Wr(i0, i1, wa, tr, ew=None, ewer=None, fl=None, er=None, co=None,
            cohi=0.02, colo=0.02):
    """ Find the rest equivalent width of a feature, and column
    density assuming optically thin.

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
    cohi : float (0.02)
      When calculating one sigma upper error and detection limit,
      increase the continuum by this fractional amount. Only used if
      fl, er and co are also given.
    colo : float (0.02)
      When calculating one sigma lower error decrease the continuum by
      this fractional amount.  Only used if fl, er and co are also
      given.
      
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
    wa1 = wa[i0:i1+1]
    if ew is None:
        wedge = find_wa_edges(wa1)
        dw = wedge[1:] - wedge[:-1]
        ew1 = dw * (1 - fl[i0:i1+1] / co[i0:i1+1]) 
        ewer1 = dw * er[i0:i1+1] / co[i0:i1+1]
        c = (1 + cohi) * co[i0:i1+1]
        ew1hi = dw * (1 - fl[i0:i1+1] / c)
        ewer1hi = dw * er[i0:i1+1] / c
        c = (1 - colo) * co[i0:i1+1]
        ew1lo = dw * (1 - fl[i0:i1+1] / c)
        ewer1lo = dw * er[i0:i1+1] / c
    else:
        ew1 = np.array(ew[i0:i1+1])
        ewer1 = np.array(ewer[i0:i1+1])

    # interpolate over bad values
    good = ~np.isnan(ew1) & (ewer1 > 0)
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
    # underestimate if saturated)
    Nmult = 1.13e20 / (tr['osc'] * tr['wa']**2)

    # 5 sigma detection limit
    detlim = np.log10( Nmult * 5*Wre )

    c = np.log10(Nmult * Wr)
    if fl is not None:
        chi = np.log10( Nmult * (Wrhi + Wrehi) )
        clo = np.log10( Nmult * (Wrlo - Wrelo) )
    else:
        chi = np.log10( Nmult * (Wr + Wre) )
        clo = np.log10( Nmult * (Wr - Wre) )

    return adict(logN=(clo,c,chi), Ndetlim=detlim, Wr=Wr, Wre=Wre, zp1=zp1,
                 ngoodpix=good.sum(), Nmult=Nmult)

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
    """ Convert temperatues in K to b parameters (km/s) for an atom
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
        filename = DATAPATH + '/linelists/atom.dat'

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
    i = 1
    while name[i] not in 'XVI':
        i += 1
    return name[:i], name[i:]

def tau_LL(logN, wa, wstart=912.):
    """ Find the optical depth at the neutral hydrogen Lyman limit.

    Parameters
    ----------
    logN : float
      log10 of neutral hydrogen column density in cm^-2.
    wa : array_like
      Wavelength in Angstroms.
    wstart : float (912.)
      Tau values at wavelengths above this are zero.

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

def calc_DLA_tau(wmin, wmax, logN=20.3, logZ=0, dv=5., atom=None):
    """ Create the optical due to absorption from a DLA.

    The DLA is at z=0. The column density and metallicity can be
    varied. Solar Abundance ratios are assumed, with most of the atoms
    in the singly ionised state

    Parameters
    ----------
    wmin, wmax : float
       Start an end of the wavelength scale (rest wavelengths).
    logZ : float (0)
       log10 of the metal abundance relative to solar. For example 0
       (the default) gives solar abundances, -1 gives 1/10th of solar.
       
    Returns
    -------
    wa, tau : ndarrays, shape (N,)
      The wavelength array and tau at each wavelength.

    """
    wa = make_constant_dv_wa_scale(wmin, wmax, dv)
    off = -12 + logN + logZ
    f26 = (
        'HI    0  0  50 0 %.2f 0' % logN,    
        'OI    0  0  50 0 %.2f 0' % (Asolar['O']  + off), 
        'SiII  0  0  50 0 %.2f 0' % (Asolar['Si'] + off - 0.05), 
        'SiIII 0  0  50 0 %.2f 0' % (Asolar['Si'] + off - 1),    
        'SiIV  0  0  50 0 %.2f 0' % (Asolar['Si'] + off - 1),    
        'FeII  0  0  50 0 %.2f 0' % (Asolar['Fe'] + off),    
        'CII   0  0  50 0 %.2f 0' % (Asolar['C']  + off - 0.05),    
        'CIII  0  0  50 0 %.2f 0' % (Asolar['C']  + off - 1),    
        'CIV   0  0  50 0 %.2f 0' % (Asolar['C']  + off - 1),
        'CaII  0  0  50 0 %.2f 0' % (Asolar['Ca'] + off),
        'AlII  0  0  50 0 %.2f 0' % (Asolar['Al'] + off - 0.05),
        'AlIII 0  0  50 0 %.2f 0' % (Asolar['Al'] + off - 1),
        'TiII  0  0  50 0 %.2f 0' % (Asolar['Ti'] + off),
        'NII   0  0  50 0 %.2f 0' % (Asolar['N']  + off),
        'ZnII  0  0  50 0 %.2f 0' % (Asolar['Zn']  + off),
        'CrII  0  0  50 0 %.2f 0' % (Asolar['Cr']  + off),
        )
    f26 = StringIO('\n'.join(f26))
    if atom is None:
        atom = readatom()

    #import pdb; pdb.set_trace()
    tau,ticks = find_tau(wa, f26, atom)
    tau += tau_LL(logN, wa, wstart=912.)
    return wa, tau, ticks

def calc_DLA_trans(wa, redshift, vfwhm, logN=20.3, logZ=0, dv=5.):
    """ Find the transmission after absrption by a DLA

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

    Returns
    -------
    transmission : ndarray, shape (N,)
      The transmission at each wavelength
    ticks : ndarray, shape (M,)
      Information for making ticks to show absorbing components.
    """
    wa1 = wa / (1 + redshift)
    lwa1, tau, ticks = calc_DLA_tau(wa1[0], wa1[-1], logN=logN, logZ=logZ,
                                    dv=dv)
    lwa = lwa1 * (1 + redshift)
    trans = convolve_psf(np.exp(-tau),  vfwhm / dv)
    trans1 = np.interp(wa, lwa, trans)
    return trans1, ticks
