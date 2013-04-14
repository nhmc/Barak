from ..convolve import *
import numpy as np

def test_convolve_psf():
    np.random.seed(117)
    a = np.random.randn(20)
    res = convolve_psf(a, 3)
    assert np.allclose(
        res, np.array(
            [-1.82810164, -0.87019123, -0.266669  , -0.03265678,  0.0983152 ,
             0.30890871,  0.53010346,  0.54824343,  0.30440922, -0.01564125,
             -0.28124608, -0.43297632, -0.33312314, -0.01804883,  0.3034258 ,
             0.43073622,  0.37775529,  0.35787026,  0.51013895,  0.81133282])
        )

def test_convolve_constant_dv():
    np.random.seed(117)
    wa = np.arange(1215, 1220, 0.05)
    fl = np.random.randn(len(wa))
    res = convolve_constant_dv(wa, fl, vfwhm=6.6)
    assert np.allclose(
        res, np.array(
            [-1.82810164, -0.49383329,  0.27038313, -0.18174949,  0.06144131,
             0.12365081,  0.79699317,  0.88245902,  0.40018728, -0.46657634,
             0.28910753, -1.26211566, -0.16543775, -0.26325761,  0.60501931,
             0.76975907,  0.12304196,  0.31886655,  0.2710288 ,  0.82700837,
             1.4031358 ,  0.32359982, -0.41666449,  0.99491426, -0.04947175,
             -0.2777515 ,  0.64690736,  0.41702481, -0.12153615, -0.52416729,
             0.74601352,  0.52626036, -0.16133486,  0.23434163,  0.20375711,
             1.65184694,  1.02838028,  0.93001476,  0.33101733, -1.00490192,
             -0.63205387,  0.28354749, -1.4381868 ,  1.17273431,  0.27584377,
             -0.27649119,  0.62589137,  0.58910533,  0.16386814,  1.30712774,
             -0.04767614,  1.25609984,  1.55498515,  0.9564274 , -0.41708657,
             0.29424205,  0.12398363,  1.00293999,  0.09602291,  0.29074619,
             0.25453467,  0.37990766, -0.05836495, -0.41131933,  0.52939114,
             0.31393051,  0.63223186,  2.07574209,  1.56515786,  0.83304359,
             2.10464305, -0.43873194,  1.81362591,  0.22960783,  0.64520274,
             0.28605202,  0.32358639, -0.56834926, -0.81643087, -1.00366126,
             -1.09714023, -1.33624775, -0.86836803, -2.15087615,  0.05409471,
             -0.91965209, -1.30392306, -0.50115069, -2.13478332, -1.61989867,
             0.2840823 ,  0.51971044,  0.13173418, -0.3143674 ,  0.2950768 ,
             -0.51319048, -0.34933121,  0.37323927,  1.74916909,  1.41902821])
        )
