""" Useful physical and mathematical values. Physical constants in
Gaussian cgs units when not indicated otherwise. From 2010 CODATA
recommended values where available (see
http://physics.nist.gov/cuu/Constants/index.html).

>>> import constants as c
>>> from math import sqrt 
>>> Planck_length = sqrt(c.hbar * c.G / c.c**3)    # cm 
>>> Planck_mass = sqrt(c.hbar * c.c / c.G)         # g
>>> Planck_time = sqrt(c.hbar * c.G / c.c**5)      # s

Constants defined:

 ======== ===================== =============== ===============================
 c         2.99792458e10        cm/s            speed of light
 G         6.67384e-8           cm^3/g/s^2      gravitational constant
 hplanck   6.6260775e-27        erg s           Planck's constant
 hbar      1.054571726e-27      erg s           1/(4*pi) * Planck's constant
 kboltz    1.3806488e-16        erg/K           Boltzmann constant
 mp        1.67261777e-24       g               proton mass
 me        9.10938291e-28       g               electron mass
 eV        1.602176565e-12      ergs            electron volt
 e         4.80320451e-10       esu             magnitude of charge on electron
 sigma     5.670373e-5          erg/s/cm^2/K^4  Stefan-Boltzmann constant
 Ryd       2.179872171e-11      ergs            Rydberg: energy needed to
                                                dissociate H atom from
                                                ground state
 Jy        1e-23                ergs/s/cm^2/Hz  Jansky
 sigmaT    6.652458734e-25      cm^2            Thomson cross section
 Mmoon     7.348e25             g               Moon mass
 Rmoon     1.7374e8             cm              Moon radius
 Mearth    5.9742e27            g               Earth mass
 Rearth    6.3781e8             cm              Earth radius
 Msun      1.989e33             g               Solar mass
 Lsun      3.90e33              erg/s           Solar luminosity
 Rsun      6.96e10              cm              Solar radius
 au        1.496e13             cm              Distance from Earth to Sun
 ly        9.4607304725808e16   cm              light year
 pc        3.08567802e18        cm              parsec
 kpc       3.08567802e21        cm              kiloparsec
 Mpc       3.08567802e24        cm              megaparsec
 yr        3.155815e7           s               year
 Gyr       3.155815e16          s               gigayear
 mu        0.62                 unitless        mean molecular weight of
                                                astrophysical gas
 mile      160934.              cm              mile
 a0        hbar**2 / me / e**2  cm              Bohr radius
 alpha     e**2 / (hbar*c)      unitless        Fine structure constant
 Ryd_Ang   h * c * 1.0e8 / Ryd  Angstroms       Rydberg in Angstroms
 c_kms     2.99792458e5         km/s            speed of light
          
 sqrt_ln2  0.832554611158       sqrt(ln(2))
 pi       

 wlya      1215.6701            Angstroms       Wavelength of HI Lya transition
 wlyb      1025.72              Angstroms       Wavelength of HI Lyb transition
  
 Ar                                             dictionary of atomic weights
 ======== ===================== =============== ===============================
"""
c       = 2.99792458e10       # cm/s            speed of light
G       = 6.67384e-8          # cm^3/g/s^2      gravitational constant
hplanck = 6.6260775e-27       # erg s           Planck's constant
hbar    = 1.054571726e-27     # erg s           1/(4*pi) * Planck's constant
kboltz  = 1.3806488e-16       # erg/K           Boltzmann constant
mp      = 1.67261777e-24      # g               proton mass
me      = 9.10938291e-28      # g               electron mass
eV      = 1.602176565e-12     # ergs            electron volt
e       = 4.80320451e-10      # esu             magnitude of charge on electron
sigma   = 5.670373e-5         # erg/s/cm^2/K^4  Stefan-Boltzmann constant
Ryd     = 2.179872171e-11     # ergs            Rydberg: energy needed to
                              #                 dissociate H atom from
                              #                 ground state
Jy      = 1e-23               # ergs/s/cm^2/Hz  Jansky
sigmaT  = 6.652458734e-25     # cm^2            Thomson cross section
Mmoon   = 7.348e25            # g               Moon mass
Rmoon   = 1.7374e8            # cm              Moon radius
Mearth  = 5.9742e27           # g               Earth mass
Rearth  = 6.3781e8            # cm              Earth radius
Msun    = 1.989e33            # g               Solar mass
Lsun    = 3.90e33             # erg/s           Solar luminosity
Rsun    = 6.96e10             # cm              Solar radius
au      = 1.496e13            # cm              Distance from Earth to Sun
ly      = 9.4607304725808e16  # cm              light year
pc      = 3.08567802e18       # cm              parsec
kpc     = 3.08567802e21       # cm              kiloparsec
Mpc     = 3.08567802e24       # cm              megaparsec
yr      = 3.155815e7          # s               year
Gyr     = 3.155815e16         # s               gigayear
mu      = 0.62                # unitless        mean molecular weight of
                              #                 astrophysical gas
mile    = 160934.             # cm              mile
a0      = hbar**2 / me / e**2 # cm              Bohr radius
alpha   = e**2 / (hbar*c)     # unitless        Fine structure constant
Ryd_Ang = hplanck * c * 1.0e8 / Ryd # Angstroms       Rydberg in Angstroms
c_kms   = 2.99792458e5        # km/s            speed of light

sqrt_ln2 = 0.832554611158     # sqrt(ln(2))
from math import pi

wlya = 1215.6701              # Angstroms       Wavelength of HI Lya transition
wlyb = 1025.72                # Angstroms       Wavelength of HI Lyb transition

# atomic weights from http://www.nist.gov/pml/data/comp.cfm
Ar = dict(H=1.00794,
          He=4.002602,
          C=12.0107,
          N=14.0067,
          O=15.9994,
          Mg=24.3050,
          Al=26.9815386,
          Si=28.0855,
          P=30.973762,
          S=32.065,
          Ca=40.078,
          Fe=55.845,
          )
