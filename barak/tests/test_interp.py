from ..interp import trilinear_interp
import numpy as np

def test_trilinear_interp():
    Nx, Ny, Nz = 10., 12., 14.
    X,Y,Z = np.mgrid[0:Nx, 0:Ny, 0:Nz]
    vals = np.cos(pi/Nx*(X-Z)) - np.sin(pi/Ny*(Y+Z))
    x0 = np.arange(Nx)
    y0 = np.arange(Ny)
    z0 = np.arange(Nz)

    N2x, N2y, N2z = 100., 101., 102.
    x1 = np.arange(N2x)* float(Nx-1) / N2x
    y1 = np.arange(N2y)* float(Ny-1) / N2y
    z1 = np.arange(N2z)* float(Nz-1) / N2z
    
    out = trilinear_interp(x1,y1,z1, x0,y0,z0,vals)

    # from ..plot import arrplot
    #arrplot(x0, y0, vals[:, :, 0])
    #arrplot(x1, y1, out[:, :, 0])

    #arrplot(y0, z0, vals[0, :, :])
    #arrplot(y1, z1, out[0, :, :])
    #pl.show()
