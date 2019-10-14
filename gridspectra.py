import scipy
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import radialProfile as rp

'''
if len(sys.argv) != 5:
    print 'Usage: ./grid2pk.py <gridfile1> <gridfile2> <boxsize> <outbasename>'
    print '       for example the command'
    print '         ./grid2pk.py grid1.bin grid2.bin 100 test'
    print '       will read in N^3 32-bit floats from grid[1,2].bin'
    print '       (with N determined from the filesize) and output the'
    print '       power spectrum into the file outbasename.npz'
    sys.exit()
'''

class gridspectra:
    def __init__(self,auto=True,boxsize=512.0,test=False):

        self.auto    = auto
        self.boxsize = boxsize
        self.test    = test
        
        self.dk  = 2*np.pi/boxsize
        self.d3k = boxsize**(-3)

        self.spectra = {}

    def loadgrid(self,gridfile):

        # Get size
        fsize   = os.stat(gridfile).st_size
        dsize   = fsize/4
        n3      = int(round(dsize**(1./3.)))
        if self.test == True: n3 = 16      
        dsize   = n3**3

        # Read in grid data
        griddata = np.fromfile(gridfile,count=dsize,dtype=np.float32)
        griddata = np.reshape(griddata,[n3,n3,n3])

        self.fsize = fsize
        self.dsize = dsize
        self.n3    = n3
        
        return griddata

    def autopower(self,grid):

        datacplx  = fftpack.fftn(grid)
        datacplx  = fftpack.fftshift(datacplx)

        p3d = np.abs(datacplx)**2

        return p3d

    def crosspower(self,grid1,grid2):

        datacplx1  = fftpack.fftn(grid1)
        datacplx1  = fftpack.fftshift(datacplx1)

        datacplx2  = fftpack.fftn(grid2)
        datacplx2  = fftpack.fftshift(datacplx2)

        p3d1 = np.abs(datacplx1)**2
        p3d2 = np.abs(datacplx2)**2
        cc3d = np.abs(datacplx1*np.conj(datacplx2))

        return p3d1,p3d2,cc3d
        
    def power(self,grid1,grid2=[]):

        if len(grid2) == len(grid1):
            
            p3d1,p3d2,cc3d = self.crosspower(grid1,grid2)

            p1d1 = rp.sphericalAverage(p3d1) / self.d3k / self.dsize**2
            p1d2 = rp.sphericalAverage(p3d2) / self.d3k / self.dsize**2
            cc1d = rp.sphericalAverage(cc3d) / self.d3k / self.dsize**2

            k1d  = np.arange(len(p1d1))+1.5 # Add 1 here for the Nyquist plane
            k1d *= self.dk

            return k1d,p1d1,p1d2,cc1d

        else:

            p3d1 = self.autopower(grid1)
            p1d1 = rp.sphericalAverage(p3d1) / self.d3k / self.dsize**2
            k1d  = np.arange(len(p1d1))+1.5 # Add 1 here for the Nyquist plane
            k1d *= self.dk

            return k1d,p1d1
                     


