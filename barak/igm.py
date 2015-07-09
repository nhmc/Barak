""" Code related to analysis of the intergalactic medium.
"""

# teff from Becker et al.
def tau_eff_Becker13(z, z0=3.5, tau0=0.751, beta=2.90, C=-0.132):
    """ Effective optical depth of neutral hydrogen.

    Parameters
    ----------
    z : float or array
      Redshift

    Returns
    -------
    tau : the effective optical depth

    Notes
    -----
    From Becker et al. 2013, section 3.2: 2013MNRAS.430.2067B
    """

    return tau0 * ((1+z) / (1+z0))**beta + C
