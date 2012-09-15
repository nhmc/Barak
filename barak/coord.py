"""Astronomical coordinate functions."""
import re
import numpy as np
from numpy.core.records import fromarrays

from math import pi, cos

DEG_PER_HR = 360. / 24.           
DEG_PER_MIN = DEG_PER_HR / 60.    
DEG_PER_S = DEG_PER_MIN / 60.     
DEG_PER_AMIN = 1./60.             
DEG_PER_ASEC = DEG_PER_AMIN / 60. 
RAD_PER_DEG = pi / 180.
DEG_PER_RAD = 180. / pi

def radec_to_xyz(ra_deg, dec_deg):
    """ Convert RA and Dec to xyz positions on a unit sphere.

    Parameters
    ----------
    ra_deg, dec_deg : float or arrays of floats, shape (N,)
         RA and Dec in degrees.

    Returns an array of floats with shape (N, 3).
    """
    ra  = np.asarray(ra_deg) * RAD_PER_DEG
    dec = np.asarray(dec_deg) * RAD_PER_DEG
    cosd = np.cos(dec)
    xyz = np.array([cosd * np.cos(ra),
                    cosd * np.sin(ra),
                    np.sin(dec)]).T

    return np.atleast_2d(xyz)

def distsq(ra1, dec1, ra2, dec2):
    """ Find the distance squared in xyz space between two RAs and
    Decs.

    Parameters
    ----------
    
    ra1, dec1 :  floats or arrays of floats, shape (N,)
    ra2, dec2 :  floats or arrays of floats, shape (M,) 

    Returns
    -------
    distance_squared: array of floats shape (N, M)
       If N or M is 1, that dimension is suppressed.
    """
    xyz1 = radec_to_xyz(ra1, dec1)
    xyz2 = radec_to_xyz(ra2, dec2)

    n = xyz1.shape[0]
    m = xyz2.shape[0]
    d2 = np.empty((n, m))
    for i in range(n):
        d2[i,:] = ((xyz1[i,:] - xyz2)**2).sum(axis=1)

    d2 = d2.squeeze()
    if len(d2.shape) == 0:
        d2 = float(d2)
    return d2

def radians_to_distsq(radians):
    """ Convert to a squared xyz separation from an angle.

    The input is the angle in radians. The conversion is done on a
    unit sphere using the cosine rule.
    """
    return 2 * (1 - np.cos(radians))

def distsq_to_radians(distsq):
    """ Convert to an angle from a squared xyz separation.

    The output angle is in radians. The conversion is done on a unit
    sphere using the cosine rule.
    """
    return np.arccos(1 - 0.5 * distsq)

def check_ra_dec(ra, dec):
    """ Check 0 <= RA < 360 and -90 <= Dec <= 90.

    Raises a ValueError outside these limits.
    
    Parameters
    ----------
    ra, dec : floats or arrays of floats
      RA and Dec in degrees.
    """
    ra = np.atleast_1d(ra)
    dec = np.atleast_1d(dec)
    msg = []
    if (ra < 0).any():
        msg.append('RA must be >= 0, %f' % ra[ra < 0][0])
    if (ra >= 360).any():
        msg.append('RA must be < 360, %f' % ra[ra >= 360][0])
    if (dec < -90).any():
        msg.append('Dec must be >= -90, %f' % dec[dec < -90][0])
    if (dec > 90).any():
        msg.append('Dec must be <= 90, %f' % dec[dec > 90][0])

    if msg:
        raise ValueError('\n'.join(msg))

def ang_sep(ra1, dec1, ra2, dec2):
    """ Returns the angular separation in degrees on the celestial
    sphere between two RA/Dec coordinates.

    Parameters
    ----------
    ra1, dec1 : floats or arrays of floats, shape (N,)
       First set of coordinates in degrees.
    ra2, dec2 : floats or arrays of floats, shape (M,) 
       Second set of coordinates in degrees.

    Returns
    -------
    separation_in_degrees : array of floats, shape (N, M)
       If N or M is 1, that dimension is suppressed.
    """
    check_ra_dec(ra1, dec1)
    check_ra_dec(ra2, dec2)
    d2 = distsq(ra1, dec1, ra2, dec2)
    return DEG_PER_RAD * distsq_to_radians(d2)

def ra_dec2s(ra, raformat='%02.0f %02.0f %06.3f'):
    ra = float(ra)
    if not (0.0 <= ra < 360.):
        raise ValueError("RA outside sensible limits: %s" % ra)

    rah, temp = divmod(ra, DEG_PER_HR)
    ram, temp = divmod(temp, DEG_PER_MIN)
    ras = temp / DEG_PER_S
    s_ra = raformat % (rah, ram, ras)

    return s_ra

def dec_dec2s(dec, decformat='%02.0f %02.0f %05.2f'):
    """ Converts decimal RA and Dec to sexigesimal.

    Returns two strings, RA and Dec.
    """
    dec = float(dec)
    if dec < 0.:
        dec *= -1.
        negdec = True
    else:  negdec = False
    # error checking
    if dec > 90.:
        raise ValueError("Dec outside sensible limits: %s" % dec)

    decd, temp = divmod(dec, 1)
    decm, temp = divmod(temp, DEG_PER_AMIN)
    decs = temp / DEG_PER_ASEC
    if negdec:
        s_dec = '-' + decformat % (decd, decm, decs)
    else:  s_dec = '+' + decformat % (decd, decm, decs)

    return s_dec

def dec2s(ra, dec):
    """ Convert an RA and Dec from degrees to sexigesimal.

    Parameters
    ----------
    ra, dec: floats or arrays of floats, shape (N,)
      The RA and Dec in degrees.

    Returns
    -------
    ra, dec: str or arrays of str, shape (N,)
      The RA and Dec in 'hour:min:s' 'deg:min:s' format.
    """
    try:
        return ra_dec2s(ra), dec_dec2s(dec)
    except TypeError:
        pass
    radec = [(ra_dec2s(r), dec_dec2s(d)) for r, d in zip(ra, dec)]
    return tuple(zip(*radec))

def ra_s2dec(ra):
    """ Converts a sexigesimal RA string to decimal.

    ra : string or sequence of three strings
      The input hour, minute and second. If a string, separators
      between hours minutes and seconds can be whitespace, colons or
      h, m. s.
    """
    if isinstance(ra, basestring):
        ra = re.sub('[:hms]', ' ', ra)
        ra = ra.split()

    rah,ram,ras = [float(item) for item in ra]
    if not 0. <= rah < 24. or not 0. <= ram <= 60. or not 0. <= ras <= 60.:
        raise ValueError('RA is outside sensible limits. RA = %s' % ra)

    d_ra = DEG_PER_HR * rah + DEG_PER_MIN * ram + DEG_PER_S * ras
    return d_ra

def dec_s2dec(dec):
    """ Converts a sexigesimal Dec string to decimal.

    The separators between deg/arcmin/arcsec can be whitespace or
    colons or d m s.
    """
    # Convert to floats, noting sign of dec
    if isinstance(dec, basestring):
        dec = re.sub('[:dms]', ' ', dec)
        dec = dec.split()
    if dec[0].lstrip()[0] == '-':
        negdec = True
    else:
        negdec = False
    decd,decm,decs = [float(item) for item in dec]
    if negdec:
        decd *= -1.
    # Error checking
    if decd > 90. or decm >= 60. or decs > 60:
        raise ValueError('Dec is outside sensible limits: Dec = %s' % dec)

    d_dec = decd + DEG_PER_AMIN * decm + DEG_PER_ASEC * decs
    if negdec:
        d_dec *= -1.

    return d_dec

def s2dec(ra, dec):
    """ Convert a sexigesimal ra and dec (or list of ras and decs) to
    decimal degrees.

    Parameters
    ----------
    ra, dec: str or arrays of str, shape (N,)
      The RA and Dec in 'hour:min:s' 'deg:min:s' format. Separators
      may be whitespace, colons, 'h', 'm', 's' or 'd'.

    Returns
    -------
    ra, dec: floats or arrays of floats, shape (N,)
      The RA and Dec in degrees.

    Examples
    --------
    >>> s2dec('02h59m00.56s', '-80d10m04.3s')
    (44.75233333333333, -80.16786111111112)

    >>> sras = ['10:12:01.25', '10:14:06.13']
    >>> sdecs =['01:01:45.65', '01:13:47.02']
    >>> ra, dec = s2dec(sras, sdecs)
    >>> print zip(ra, dec)
    [(153.00520833333334, 1.0293472222222222),
    (153.52554166666667, 1.229727777777778)]
    """

    if isinstance(ra, basestring):
        return ra_s2dec(ra), dec_s2dec(dec)

    radec = [(ra_s2dec(r), dec_s2dec(d)) for r, d in zip(ra, dec)]
    return tuple(map(np.array, zip(*radec)))

def match(ra1, dec1, ra2, dec2, tol, allmatches=False):
    """ Given two sets of numpy arrays of ra,dec and a tolerance tol,
    returns an array of indices and separations with the same length
    as the first input array.

    If an index is > 0, it is the index of the closest matching second
    array element within tol arcsec.  If it's -1, then there was no
    matching ra/dec within tol arcsec.

    If allmatches = True, then for each object in the first array,
    return the index and separation of everything in the second array
    within the search tolerance, not just the closest match.

    See Also
    --------
    indmatch, unique_radec

    Notes
    -----
    To get the indices of objects in ra2, dec2 without a match, use

    >>> imatch = match(ra1, dec1, ra2, dec2, 2.)
    >>> inomatch = numpy.setdiff1d(np.arange(len(ra2)), set(imatch))
    """
    
    ra1,ra2,dec1,dec2 = map(np.asarray, (ra1, ra2, dec1, dec2))

    abs = np.abs

    isorted = ra2.argsort()
    sdec2 = dec2[isorted]
    sra2 = ra2[isorted]

    LIM = tol * DEG_PER_ASEC

    match = []
    # use mean dec, assumes decs similar
    decav = np.mean(sdec2.mean() + dec1.mean())
    RA_LIM = LIM / cos(decav * RAD_PER_DEG)

    for ra,dec in zip(ra1,dec1):
        i1 = sra2.searchsorted(ra - RA_LIM)
        i2 = i1 + sra2[i1:].searchsorted(ra + RA_LIM)
        #print i1,i2
        close = []
        for j in xrange(i1,i2):
            if abs(dec - sdec2[j]) > LIM:
                continue
            else:
                # if ras and decs are within LIM arcsec, then
                # calculate actual separation:
                disq = ang_sep(ra, dec, sra2[j], sdec2[j])
                close.append((disq, j))

        close.sort()
        if not allmatches:
            # Choose the object with the closest separation inside the
            # requested tolerance, if one was found.
            if len(close) > 0:
                min_dist, jmin = close[0]
                if min_dist < LIM:
                    match.append((isorted[jmin], min_dist))
                    continue
            # otherwise no match
            match.append((-1,-1))
        else:
            # append all the matching objects
            jclose = []
            seps = []
            for dist,j in close:
                if dist < LIM:
                    jclose.append(j)
                    seps.append(dist)
                else:
                    break
            match.append(fromarrays([isorted[jclose], seps],
                                    dtype=[('ind','i8'), ('sep','f8')]))

    if not allmatches:
        # return both indices and separations in a recarray
        temp = np.rec.fromrecords(match, names='ind,sep')
        # change to arcseconds
        temp.sep *= 3600.
        temp.sep[temp.sep < 0] = -1.
        return temp
    else:
        return match

def indmatch(ra1, dec1, ra2, dec2, tol):
    """ Finds objects in ra1, dec1 that have a matching object in
    ra2, dec2 within tol arcsec.

    Parameters
    ----------
    ra1, dec1 : arrays of floats, shape (N,)
      First list of coordinates in degrees.
    ra2, dec2 : arrays of floats, shape (M,)
      Second list of coordinates in degrees.

    Returns
    -------
    i1 : arrays of int, shape (P,)
      `i1` are the indices into ra1,dec1 that have matches in the ra2,
      dec2. `i2` are the indices into ra2,dec2 giving the matching objects.

    See Also
    --------
    match, unique_radec
    """
    m = match(ra1, dec1, ra2, dec2, tol)
    c = m.ind > -1
    i1 = c.nonzero()[0]
    i2 = m.ind[c]
    return i1, i2
    
def unique_radec(ra, dec, tol):
    """ Find unique ras and decs in a list of coordinates.

    RA and Dec must be arrays of the same length, and in degrees.

    tol is the tolerance for matching in arcsec. Any coord separated by
    less that this amount are assumed to be the same.

    Returns
    -------
    ind1 : ndarray of ints, shape (N,)
      Indices of the first occurence of a unique coordinate in the
      input array.
    ind2 : list of int arrays, length N
      Indices of all coords that were matched to a given unique coordinate.

    See Also
    --------
    indmatch, match

    The matching algorithm is confusing, but hopefully correct and not too
    slow. Potential for improvement...
    """
    matches = match(ra, dec, ra, dec, tol, allmatches=True)
    imatchflat = []
    for m in matches:
        imatchflat.extend(m.ind)

    inomatch = np.setdiff1d(np.arange(len(ra)), list(set(imatchflat)))

    assert len(inomatch) == 0
    # Indices giving unique ra, decs
    iunique = []
    # Will be same length as iunique. Gives all indices in original
    # coords that are matched to each unique coord.
    iextras = []
    assigned = set()
    for j,m in enumerate(matches):
        if not (j % 1000):
            print j
        # get the lowest index in this group
        isort = sorted(m.ind)
        ilow = isort[0]
        if ilow not in assigned:
            iunique.append(ilow)
            assigned.add(ilow)
            iextras.append([ilow])
            # assign any extra indices to this unique coord.
            for i in isort[1:]:
                # check not already been assigned to another coord
                if i not in assigned:
                    iextras[-1].append(i)
                    assigned.add(i)

    return np.array(iunique), iextras
