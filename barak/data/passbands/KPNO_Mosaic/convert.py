from astro.io import writetxt
from astro.utilities import adict
from glob import glob

def readfilt(name):
    fh = open(name)
    rows = fh.readlines()
    fh.close()
    if rows[0].startswith('#'):
        print name, 'already converted, skipping'
        return None
    header = []
    i = 0
    while not rows[i].startswith('*****'):
        header.append('# ' + rows[i].strip())
        i += 1
     
    i +=1
    wa, tr = [], []
    for r in rows[i:]:
        if not r.strip():
            continue
        w, t = map(float, r.split())
        wa.append(w)
        tr.append(t / 100.)
    return adict(wa=wa, tr=tr, hd='\n'.join(header) + '\n')

if 1:

    names = glob('*.txt')
    for name in names:
        f = readfilt(name)
        if f is None:
            continue

        writetxt(name, [f.wa, f.tr], header=f.hd)
