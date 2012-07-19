from ..spec import *
from ..utilities import get_data_path

DATAPATH = get_data_path()

def test_find_wa_edges():
    assert np.allclose(find_wa_edges([1, 2.1, 3.3, 4.6]),
                       [0.45,  1.55,  2.7,   3.95,  5.25])

def test_make_wa_scale():
    wa = make_wa_scale(40, 1, 5)
    assert np.allclose(wa, [40., 41., 42., 43., 44.])

    wa = make_wa_scale(3.5, 1e-3, 5, constantdv=True)
    assert np.allclose(wa, [3162.278, 3169.568, 3176.874, 3184.198, 3191.537])

def test_Spectrum_init():
    sp = Spectrum(wstart=4000, dw=1, npts=500)
    assert np.allclose(sp.fl, 0)
    
    sp = Spectrum(wstart=4000, dv=60, npts=500)
    assert np.allclose(sp.dw, 8.692773e-05)
    
    sp = Spectrum(wstart=4000, wend=4400, npts=500)
    assert np.allclose(sp.dw, 0.80160)
    
    wa = np.linspace(4000, 5000, 500)
    fl = np.ones(len(wa))
    sp = Spectrum(wa=wa, fl=fl)
    assert np.allclose(sp.dw, 2.004008)
    
    sp = Spectrum(CRVAL=4000, CRPIX=1, CDELT=1, fl=np.ones(500))
    assert np.allclose(sp.dw, 1), np.allclose(sp.wa[0],4000)

def test_Spectrum_stats():
    wa = np.linspace(10,20,10)
    np.random.seed(77)
    fl = np.random.randn(len(wa)) + 10
    er = np.ones(len(wa))
    sp = Spectrum(wa=wa, fl=fl, er=er)
    data = sp.stats(11, 18)
    assert np.allclose(data, [9.66873, 0.98909, 1.0, 9.77542])

def test_read():
    sp = read(DATAPATH + 'tests/HE0940m1050m.txt.gz')
    sp = read(DATAPATH + 'tests/spSpec-52017-0516-139.fit.gz')
    sp = read(DATAPATH + 'tests/runA_h1_100.txt.gz')
    assert np.allclose(sp.fwhm, 6.6)


def test_rebin():
    wa = np.linspace(11,20,10)
    np.random.seed(77)
    fl = np.random.randn(len(wa)) + 10
    er = np.ones(len(wa))
    er[3] = 0
    sp = Spectrum(wa=wa, fl=fl, er=er)
    rsp = sp.rebin(wstart=sp.wa[0], wend=sp.wa[-1], dw=1.5*sp.dw)

    assert np.allclose(rsp.fl, [10.3119,10.0409,9.94336,9.2459,9.5919,9.9963])
    assert np.allclose(rsp.er, [0.89443,0.81650,1.4142,0.8165,0.8165,0.8165])
    assert np.allclose(rsp.wa, [11., 12.5, 14., 15.5, 17., 18.5])


def test_combine():
    wa = np.linspace(11,20,10)
    np.random.seed(77)
    fl1 = np.random.randn(len(wa)) + 10
    fl2 = np.random.randn(len(wa)) + 10
    er1 = np.ones(len(wa))
    er1[7] = -1
    er2 = np.ones(len(wa))
    er2[7] = -1
    er2[3] = -1
    sp1 = Spectrum(wa=wa, fl=fl1, er=er1)
    sp2 = Spectrum(wa=wa, fl=fl2, er=er2)
    csp = combine([sp1,sp2])

    assert np.allclose(
        csp.er[~np.isnan(csp.er)],
        [0.70711, 0.70711, 0.70711, 1.,
        0.70711, 0.70711, 0.70711, 0.70711, 0.70711])

    assert np.allclose(
        csp.fl[~np.isnan(csp.fl)],
        [ 10.2652, 10.9127, 9.1856,
          10.4078, 9.9137, 9.8614, 9.4947, 10.7472, 9.5573])


def test_air2vac_vac2air():
    assert np.allclose(vac2air_Ciddor(air2vac_Ciddor([2000, 80000])),
                       [2000, 80000], rtol=1e-9)
    assert np.allclose(vac2air_Morton(air2vac_Morton([2000, 80000])),
                       [2000, 80000], rtol=1e-9)
    assert np.allclose(vac2air_Ciddor(air2vac_Ciddor(5000)), 5000, rtol=1e-9)
    assert np.allclose(vac2air_Morton(air2vac_Morton(5000)), 5000, rtol=1e-9)
