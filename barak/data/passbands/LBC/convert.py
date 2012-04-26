from glob import glob

names = glob('*.txt')

for n in names:
    b = readtxt(n, names='wa,tr')
    isort = b.wa.argsort()
    b.wa = b.wa[isort]
    b.tr = b.tr[isort]
    b.wa *= 10.
    b.tr /= 100.
    plot(b.wa, b.tr, label=n)
    writetxt(n, b)

pl.legend()
