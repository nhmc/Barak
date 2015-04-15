from ..extinction import *
import numpy as np

def basic_tests():
    wa = np.arange(900, 22000, 5)
    MW_Cardelli89(wa, EBmV=0.05)
    starburst_Calzetti00(wa, EBmV=0.05)
    # test regime in which interpolation from MW to Calzetti occurs
    starburst_Calzetti00([3000], EBmV=0.05)
    LMC_Gordon03(wa, EBmV=0.05)
    LMC2_Gordon03(wa, EBmV=0.05)
    SMC_Gordon03(wa, EBmV=0.05)
