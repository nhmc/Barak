import astro.spec
from glob import glob

names = sorted(glob('*.fits'))

for n in names:
    s = astro.spec.read(n)
    writetabfits(n,np.rec.fromarrays([s.wa,s.fl,s.er],names='wa,fl,er'),overwrite=1)
