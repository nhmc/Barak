from ..coord import *
from ..utilities import get_data_path

DATAPATH = get_data_path()

def test_angsep():
    np.allclose(ang_sep(2, 0, 4, 0), 2.)
    np.allclose(ang_sep(359, 0, 1, 0), 2.)
    np.allclose(ang_sep(0, 20, 0, -10), 30)
    np.allclose(ang_sep(7, 20, 8, 40), 20.018358)
    np.allclose(ang_sep(7, 20, 250, -50.3), 122.388401)

def test_dec2s():
    assert dec2s(156.1125638,-10.12986) == ('10 24 27.015', '-10 07 47.50')
    assert dec2s(0.0,-90.0) == ('00 00 00.000', '-90 00 00.00')

def test_s2dec():
    assert s2dec('00:00:00', '90:00:00') == (0.0, 90.0)
    temp = np.array(s2dec('10 24 27.015', '-10 07 47.50'))
    reference = np.array([156.1125625,-10.129861111111111])
    assert np.all(temp - reference < 1.e-10)
