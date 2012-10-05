"""Contains a class, VpfitModel, useful for parsing f26 and fort.13
files, and writing out fort.13 files.
"""
import os
import numpy as np
from textwrap import wrap
from constants import c_kms

# the data types of the lines and regions numpy arrays:

len_filename = 150

dtype_lines = [('name', 'S6'),
               ('z', 'f8'),
               ('zpar', 'S2'),
               ('b', 'f8'),
               ('bpar', 'S2'),
               ('logN', 'f8'),
               ('logNpar', 'S2'),
               ('zsig', 'f8'),
               ('bsig', 'f8'),
               ('logNsig', 'f8')]

dtype_regions = [('filename', 'S%i' % len_filename),
                 ('num', 'S2'),
                 ('wmin', 'f8'),
                 ('wmax', 'f8'),
                 ('resolution', 'S100')]

def parse_entry(entry):
    """ Separates an entry into a numeric value and a tied/fixed
    parameter, if present.
    """
    if entry.startswith('nan'):
        val = float(entry[:3])
        par = entry[3:]
    else:
        i = -1
        while not entry[i].isdigit():  i -= 1
        if i != -1:
            val = float(entry[:i+1])
            par = entry[i+1:]
        else:
            val = float(entry)
            par = ''

    return val,par

def parse_lines(params):
    """ Separates the parameters from their tied/fixed/special
    characters.
    """
    #print params
    temp = []
    for name,z,b,logN,zsig,bsig,logNsig in params:
        z, zpar = parse_entry(z)
        b, bpar = parse_entry(b)
        logN, logNpar = parse_entry(logN)
        try:
            zsig = float(zsig)
        except ValueError:
            zsig = -1
        try:
            bsig = float(bsig)
        except ValueError:
            bsig = -1
        try:
            logNsig = float(logNsig)
        except ValueError:
            logNsig = -1

        temp.append((name,z,zpar,b,bpar,logN,logNpar,
                     zsig,bsig,logNsig))

    temp = np.rec.fromrecords(temp, dtype=dtype_lines)
    return temp

def parse_regions(rows, res=None):
    """ Parses the region information from a f26 or fort.13 file. """
    if res is None:
        res = ''
    out = None
    rinfo = []
    for row in rows:
        r = row.split('!')[0].lstrip().lstrip('%%').split()
        nitems = len(r)
        r[2] = float(r[2])
        r[3] = float(r[3])
        if nitems == 4:
            rinfo.append(tuple(r + [res]))
        elif nitems > 4:
            r = r[:4] + [' '.join(r[4:])]
            rinfo.append(tuple(r))
        else:
            raise Exception('bad format in fitting regions:\n %s' % row)

    if rinfo:
        out = np.rec.fromrecords(rinfo, dtype=dtype_regions)
    return out


def sumlines(lines):
    """ Given several lines (record array), returns them in the vpfit
    summed format. """

    summedlines = lines.copy()

    logNtots = np.log10(np.sum(10**lines.logN))

    for i,logNtot in enumerate(logNtots):
        if i == 0:
            summedlines[i].logN = logNtot
            #summedlines[i].logNstr = '%7.4f' % logNtot
        summedlines[i].logNpar = 'w'
    
    return summedlines

class VpfitModel(object):
    """ Holds all the info about a vpfit model.  Can write out the
    model as a fort.13 or fort.26 style file.
    """
    def __init__(self, names=None, logN=None, z=None, b=None,
                 zpar=None, bpar=None, logNpar=None,
                 filenames=None, wmin=None, wmax=None, res=None, num=None):
        if None in (names,logN,z,b):
            self.lines = None              # record array
        else:
            assert len(z) == len(logN) == len(b) == len(names)
            ncomp = len(z)
            if zpar is None:  zpar = [''] * ncomp
            if bpar is None:  bpar = [''] * ncomp
            if logNpar is None:  logNpar = [''] * ncomp
            zsig = [-1] * ncomp
            bsig = [-1] * ncomp
            logNsig = [-1] * ncomp
            temp = np.rec.fromarrays([names,z,zpar,b,bpar,logN,logNpar,zsig,
                                     bsig,logNsig], dtype=dtype_lines)
            self.lines = temp
        if None in (filenames, wmin, wmax):
            self.regions = None            # record array
        else:
            if res is None:
                res = [''] * len(filenames)
            if num is None:
                num = ['1'] * len(filenames)
            assert all((len(n) < len_filename) for n in filenames)
            temp = np.rec.fromarrays([filenames,num,wmin,wmax,res],
                                    dtype=dtype_regions)
            self.regions = temp
        self.stats = None

    def __repr__(self):
        temp = ', '.join(sorted(str(attr) for attr in self.__dict__ if not str(attr).startswith('_')))
        return 'VpfitModel(%s)' % '\n      '.join(wrap(temp, width=69))

    def writef26(self,filename, write_regions=True):
        """ Writes out a f26 style file."""
        temp = []
        if write_regions and self.regions is not None:
            for r in self.regions:
                temp.append('%%%% %(filename)s  %(num)s  %(wmin)7.2f '
                            '%(wmax)7.2f  %(resolution)s\n' % r)
        if self.lines is not None:
            for line in self.lines:
                temp.append('   %(name)s     %(z)11.8f%(zpar)-2s '
                            '%(zsig)11.8f %(b)6.2f%(bpar)-2s %(bsig)6.2f '
                            '%(logN)7.4f%(logNpar)-2s %(logNsig)7.4f\n' % line)
        open(filename,'w').writelines(temp)

    def writef13(self, filename, write_regions=True):
        """ Writes out a fort.13 style file. """
        # The whitespace is important if the f13 files are to be read
        # by vpguess - don't change it!
        temp = []
        if write_regions:
            temp.append('   *\n')
            if self.regions is not None:
                for r in self.regions:
                    temp.append('%(filename)s  %(num)s  %(wmin)7.2f '
                                '%(wmax)7.2f  %(resolution)s\n' % r)
            temp.append('  *\n')
        if self.lines is not None:
            for line in self.lines:
                temp.append('   %(name)s     %(logN)7.4f%(logNpar)-2s  '
                            '%(z)11.8f%(zpar)-2s  %(b)6.2f%(bpar)-2s '
                            '0.00   0.00E+00  0\n' % line)
        open(filename,'w').writelines(temp)

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

def readf26(fh, res=None):
    """ Reads a f26 style file and returns a VpfitModel object. If the
    keyword res is given, this string provides the resolution
    information for the spectra fitted.

    For example: res='vsig=69.0'
    """
    if isinstance(fh, basestring):
        fh = open(fh)
    f = fh.readlines()
    fh.close()
    vp = VpfitModel()
    if len(f) == 0:
        #print filename, 'is empty'
        return None
    f = [r for r in f if
         not r.lstrip().startswith('!') or 'Stats' in r]
    regionrows = [r for r in f if r.lstrip().startswith('%%')]
    ionrows = [r for r in f if '%%' not in r and
               'Stats' not in r and r.lstrip()]
    keys = 'iterations nchisq npts dof prob ndropped info'.split()
    statrow = [row for row in f if 'Stats' in row]
    if statrow:
        if statrow[0].split()[-1] == 'BAD':
            status = 'BAD'
        else:
            status = 'OK'
        vals = statrow[0].split()[2:8] + [status]
        vp.stats = dict(zip(keys,vals))
    elif ionrows:
        # older style f26 file
        stat = ionrows[0]
        status = ('BAD' if stat.split()[-1] == 'BAD' else 'OK')
        vals = [stat[66:71], stat[71:85], stat[85:90], stat[90:95],
                stat[95:102], stat[102:107], status]
        vp.stats = dict(zip(keys,vals))

    vp.regions = parse_regions(regionrows,res=res)
    #print vp.regions,'\n\n\n'
    #vp.filename = filename
    if len(ionrows) == 0:
        return vp

    ionrows = [r.lstrip() for r in ionrows]
    param = []
    molecule_names = set(('H2J0 H2J1 H2J2 H2J3 H2J4 H2J5 H2J6 '
                          'COJ0 COJ1 COJ2 COJ3 COJ4 COJ5 COJ6 '
                          'HDJ0 HDJ1 HDJ2').split())
    for r in ionrows:
        if 'nan' in r:
            i = r.index('nan')
            param.append([r[:i]] + r[i:].split())
            continue
        if r[:4] in molecule_names:
            i = 4
        else:
            i = 0
            while not r[i].isdigit() and r[i] != '-':
                i += 1
        param.append([r[:i]] + r[i:].split())
            
    param = [[p[0],p[1],p[3],p[5],p[2],p[4],p[6]] for p in param]
    vp.lines = parse_lines(param)

    return vp


def readf13(filename, read_regions=True, res=None):
    """ Reads a fort.13 style file. """
    fh = open(filename)
    f = fh.readlines()
    fh.close()
    if len(f) == 0:
        #print filename, 'is empty'
        return None
    f = [row.lstrip() for row in f[1:]]      # skip past first line with *
    isep = [row[0] for row in f].index('*')  # find separating *
    vp = VpfitModel()
    if read_regions:
        vp.regions = parse_regions([row for row in f[:isep]],res=res)
    param  = [[row[:5]] + row[5:].split() for row in f[isep+1:]]
    param = [[p[0],p[2],p[3],p[1],-1,-1,-1] for p in param]
    vp.lines = parse_lines(param)
    vp.stats = None
    vp.filename = filename

    return vp

def calc_Ntot(f26name, trans=None):
    """ Calculate the total column density in f26-style file 

    Parameters
    ----------
    f26name : str
      f26 filename.
    trans : str (optional)
      Transition name ('Mg' for example). By default all column
      density entries are used.
    
    Returns
    -------
    logNtot : float
      Log10 of the total column denisty
    """
    f26 = readf26(f26name)
    logN = f26.lines.logN
    sig = f26.lines.logNsig
    
    if trans is not None:
        cond = f26.lines.name == trans
        logN = f26.lines.logN[cond]
        sig = f26.lines.logNsig[cond]

    Ntot = np.sum(10**logN)
    Nmin = np.sum(10**(logN - sig))
    Nmax = np.sum(10**(logN + sig))
    return np.log10(Ntot), np.log10(Nmin), np.log10(Nmax)
        
def calc_v90(vp, plot=False, z0=None,
             wav0=1215.6701, osc=0.4164, gam=6.265e8):
    """ For a vp model, we want to calculate the velocity width that
    contains 90% of the the total optical depth at the lya line (or
    perhaps it is the same regardless of which transition I take?) v_90
    is defined in Prochaska and Wolfe 1997.

    At the moment it guesses how big a velocity range it has to
    calculate the optical depth over - a bit dodgy"""
    lines = vp.lines
    #print 'calculating for %s' % lines
    # work in velocity space
    z = lines.z
    if z0 is None:  z0 = np.median(z)
    vel = (z - z0) / (1 + z0) * c_kms
    # calculate the optical depth as a function of velocity, 500 km/s
    # past the redmost and bluemost components - hopefully this is far
    # enough (maybe not for DLAs?)
    dv = 0.5
    vhalf = (vel.max() - vel.min())/2. + 300
    v = np.arange(-vhalf, vhalf + dv, dv)
    tau = np.zeros(len(v))
    for line,vline in zip(lines,vel):
        if line['logN'] > 21.0:
            print ('very (too?) high logN: %s' % line['logN'])
            print ('returning width of -1')
            return -1.
        temptau = calctau(v - vline, wav0, osc, gam, line['logN'],
                          btemp=line['b'])
        tau += temptau
        #pl.plot(v,tau,'+-')
        #raw_input('N %(logN)f b %(b)f enter to continue' % line)

    # integrate over the entire v range to calculate integral of tau wrt v.
    sumtaudv = np.trapz(tau,dx=dv)
    lenv = len(v)
    # starting from the left v edge, increase v until int from left
    # edge to v gives 5% of total integral
    sum5perc = sumtaudv / 20.
    sumtau = 0.
    i = 0
    while (sumtau < sum5perc):
        i += 1
        sumtau = np.trapz(tau[:i])
        if i == lenv:
            raise Exception('Problem with velocity limits!')
    vmin = v[i-1]
    # Do the same starting from the right edge.
    sumtau = 0
    i = -1
    while (sumtau < sum5perc):
        sumtau = np.trapz(tau[i:])
        i -= 1
        if -i == lenv:
            raise Exception('Problem with velocity limits!')
    vmax = v[i+1]
    # Difference between the two is v_90
    v90 = vmax - vmin
    if plot:
        pl.plot(v,tau,'+-')
        pl.vlines((vmin,vmax),0,tau.max())
    #raw_input('Enter to continue...')
    return v90

def make_rdgen_input(specfilename, filename, wmin=None, wmax=None):
    temp = ('rd %(specfilename)s\n'
            'ab\n'
            '\n'
            '\n'
            '\n'
            '%(wmin)s %(wmax)s\n'
            'qu\n'  % locals() )
    fh = open(filename,'w')
    fh.write(temp)
    fh.close()

def make_autovpin_input(specfilename, filename):
    temp = ('%(specfilename)s\n'
            '\n'
            '\n'
            '\n'
            '\n'
            '\n' % locals() )
    fh = open(filename,'w')
    fh.write(temp)
    fh.close()
