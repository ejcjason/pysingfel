import numba as nb
import math
from numba import jit, guvectorize, float64, c16
from scipy.interpolate import CubicSpline

from pysingfel.detector import *
from pysingfel.particle import *


def calculate_Thomson(ang):
    # Should fix this to accept angles mu and theta
    re = 2.81793870e-15  # classical electron radius (m)
    P = (1 + np.cos(ang)) / 2.
    return re**2 * P  # Thomson scattering (m^2)


def calculate_atomicFactor(particle, detector):
    """
    particle.ffTable contains atomic form factors against particle.qSample.
    Here the atomic form factors related to the wavevector of each pixel of
    the detector is calculated through interpolation to the ffTable data.

    Return:
        f_hkl: atomic form factors related to each pixel of detector for each atom type.
    """
    f_hkl = np.zeros((detector.py, detector.px, particle.numAtomTypes))
    q_mod_Bragg = detector.q_mod * 1e-10/2.

    for atm in range(particle.numAtomTypes):
        cs = CubicSpline(particle.qSample, particle.ffTable[atm, :])  # Use cubic spline
        f_hkl[:, :, atm] = cs(q_mod_Bragg)  # interpolate
    return f_hkl

#def Phase(atomPos,  q_xyz, angles):
    #for ix in range(q_xyz.shape[0]):
        #for iy in range(q_xyz.shape[1]):
            #for icart in range(3):
                #re[ix, iy] += atomPos[icart] * q_xyz[ix, iy, icart]


# ~20% improved wrt original implementation
#@jit
#def cal(f_hkl, atomPos, q_xyz, xyzInd):
    #F_re = np.zeros_like(q_xyz[:, :, 0], dtype=nb.double)
    #F_im = np.zeros_like(q_xyz[:, :, 0], dtype=nb.double)
    #for atm in range(atomPos.shape[0]):
        #for ix in range(F_re.shape[0]):
            #for iy in range(F_re.shape[1]):
                #angle = 2.*np.pi * ( atomPos[atm,0] * q_xyz[ix, iy, 0] +  atomPos[atm,1] * q_xyz[ix, iy, 1] + atomPos[atm,2] * q_xyz[ix, iy, 2])
                #f = f_hkl[ix, iy, xyzInd[atm]]
                #F_re[ix, iy] +=  np.cos(angle) * f
                #F_im[ix, iy] +=  np.sin(angle) * f

    #for ix in range(F_re.shape[0]):
        #for iy in range(F_re.shape[1]):
            #F_re[ix, iy] = F_re[ix, iy]**2 + F_im[ix, iy]**2

    #return F_re

# >20% improved wrt original implementation
@guvectorize(["(float64[:,:,:], float64[:,:,:], float64[:,:], int32[:], float64[:,:])"],
        "(n,m,l),(n,m,k),(j,k),(j) -> (n,m)", target="parallel")
def cal(f_hkl, q_xyz, atomPos, xyzInd, F):

    F_im = np.zeros_like(F)

    for atm in range(atomPos.shape[0]):
        for ix in range(F.shape[0]):
            for iy in range(F.shape[1]):
                angle = 2.*np.pi * ( atomPos[atm,0] * q_xyz[ix, iy, 0] +  atomPos[atm,1] * q_xyz[ix, iy, 1] + atomPos[atm,2] * q_xyz[ix, iy, 2])
                f = f_hkl[ix, iy, xyzInd[atm]]
                F[ix, iy] +=  np.cos(angle) * f
                F_im[ix, iy] +=  np.sin(angle) * f

    for ix in range(F.shape[0]):
        for iy in range(F.shape[1]):
            F[ix, iy] = F[ix, iy]**2+ F_im[ix, iy]**2

#@cuda.jit
#def cal(f_hkl, q_xyz, atomPos, xyzInd, F_re, F_im):

    ## Get a 2D CUDA grid.
    #ix, iy = cuda.grid(2)

    #if ix < F_re.shape[0] and iy < F_im.shape[1]:
    #for atm in range(atomPos.shape[0]):
        ##for ix in range(F.shape[0]):
            ##for iy in range(F.shape[1]):
                #angle = 2.*np.pi * ( atomPos[atm,0] * q_xyz[ix, iy, 0] +  atomPos[atm,1] * q_xyz[ix, iy, 1] + atomPos[atm,2] * q_xyz[ix, iy, 2])
                #f = f_hkl[ix, iy, xyzInd[atm]]
                #F_re[ix, iy] +=  np.cos(angle) * f
                #F_im[ix, iy] +=  np.sin(angle) * f

#@cuda.jit
#def modulus_square(re, im):

    #ix, iy = cuda.grid(2)

    #if ix < F_re.shape[0] and iy < F_im.shape[1]:
       #F_re[ix,iy] = F_re[ix, iy]**2+ F_im[ix, iy]**2



def calculate_molecularFormFactorSq(particle, detector):
    """
    Calculate molecular form factor for each pixel of detector to get diffraction pattern.
    Sum over all atoms with the right phase factor.
    See https://www.nature.com/article-assets/npg/srep/2016/160425/srep24791/extref/srep24791-s1.pdf
    for more details about the derivation (equ. 13 is used for calculation here).
    """
    f_hkl = calculate_atomicFactor(particle, detector)
    s = particle.SplitIdx
    xyzInd = np.zeros(s[-1], dtype=np.int32)
    detector_shape = detector.q_xyz.shape
    F_sq = np.zeros((detector_shape[0], detector_shape[1]), dtype=np.float64)
    #F_im = np.zeros_like(F_re)

    for i in range(len(s)-1):
        xyzInd[s[i]:s[i+1]] = i

    ## Initialize cuda device.
    #threads_per_block = (32,32)
    #blocks_per_grid = []
    #for i,dimension in enumerate(detector_shape):
        #blocks_per_grid.append( int(math.ceil(dimension / threads_per_block[i])) )

    #cal[blocks_per_grid, threads_per_block](f_hkl, detector.q_xyz, particle.atomPos, xyzInd, F_re, F_im)
    cal(f_hkl, detector.q_xyz, particle.atomPos, xyzInd, F_sq)

    return F_sq

def calculate_compton(particle, detector):
    """
    Calculate the contribution to the diffraction pattern from compton scattering.
    """
    half_q = detector.q_mod * 1e-10/2.
    cs = CubicSpline(particle.comptonQSample, particle.sBound)
    S_bound = cs(half_q)
    if isinstance(particle.nFree, (list, tuple, np.ndarray)):
        # if iterable, take first element to be number of free electrons
        N_free = particle.nFree[0]
    else:
        # otherwise assume to be a single number
        N_free = particle.nFree
    Compton = S_bound + N_free
    return Compton
