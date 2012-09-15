""" Photometry-based tools.
"""

def UBVRI_to_ugriz(U, BmV, UmB, RmI):
    """ Conversion from Fukugita et al. 1996."""
    g = V + 0.56 * BmV - 0.12
    r = V - 0.49 * BmV + 0.11
    umg = 1.38*UmB + 1.14
    u = umg + g
    #gmr = 1.05*(B-V) - 0.23
     
    if R - I  < 1.15:
        rmi = 0.98* RmI - 0.23
    else:
        rmi = 1.40* RmI - 0.72

    i = r - rmi

    if R - I  < 1.65:
        rmz = 1.59* RmI - 0.40
    else:
        rmz = 2.64* RmI - 2.16 

    z = r - rmz
     
    return u,g,r,i,z
