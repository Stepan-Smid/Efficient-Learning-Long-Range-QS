import numpy as np
import math
import cmath

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.tools.fit import fit_with_sum_of_exp, sum_of_exp
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import MPOModel, CouplingMPOModel, CouplingModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.tools.params import asConfig, get_parameter



class HeisenbergChainOpenBC(MPOModel,NearestNeighborModel):
# separate from periodic BC, as one can use NearestNeighborModel for this

    def __init__(self, model_params):
        n = model_params.get('L',8)
        J = model_params.get('JParams',0.5+np.zeros(n-1))
        
        site = SpinHalfSite(conserve = None, sort_charge = None)
        lat = Chain(n, site, bc_MPS="finite", bc="open")
        
        Sigmax, Sigmay, Sigmaz, Id = site.Sigmax, site.Sigmay, site.Sigmaz, site.Id
        
        
        # this sets up the matrices of the MPO
        size = 5
                            
        grids = []
        
        for i in range(n):
           
            grid = [[None for _ in range(size)] for _ in range(size)]
            
            grid[0][0] = 1*Id
            grid[size-1][size-1] = 1*Id
            grid[0][size-1] = 0*Id
            
            if i != n-1:
            
                grid[0][1] = J[i]*Sigmax
                grid[0][2] = J[i]*Sigmay
                grid[0][3] = J[i]*Sigmaz

            grid[1][size-1] = 1*Sigmax
            grid[2][size-1] = 1*Sigmay
            grid[3][size-1] = 1*Sigmaz
            
            grids.append(grid)
                            
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)
        

class HeisenbergChainFromGrid(MPOModel):

    def __init__(self, model_params):
        n = model_params.get('L',8)
        J = model_params.get('JParams',0.5+np.zeros(n))
        
        site = SpinHalfSite(conserve = None, sort_charge = None)
        lat = Chain(n, site, bc_MPS="finite", bc="open")
        
        Sigmax, Sigmay, Sigmaz, Id = site.Sigmax, site.Sigmay, site.Sigmaz, site.Id
        
        
        # this sets up the matrices of the MPO
        size = 8
                            
        grids = []
        
        for i in range(n):
        
            k = 0
            if i ==  0:
                k = J[n-1]
            if i == n-1:
                k = 1
           
            grid = [[None for _ in range(size)] for _ in range(size)]
            
            grid[0][0] = 1*Id
            grid[size-1][size-1] = 1*Id
            grid[0][size-1] = 0*Id
           
            grid[1][1] = 1*Id
            grid[2][2] = 1*Id
            grid[3][3] = 1*Id
                        
            grid[0][1] = k*Sigmax
            grid[0][2] = k*Sigmay
            grid[0][3] = k*Sigmaz
            
            grid[1][size-1] = k*Sigmax
            grid[2][size-1] = k*Sigmay
            grid[3][size-1] = k*Sigmaz
            
            grid[0][4] = J[i]*Sigmax
            grid[0][5] = J[i]*Sigmay
            grid[0][6] = J[i]*Sigmaz

            grid[4][size-1] = 1*Sigmax
            grid[5][size-1] = 1*Sigmay
            grid[6][size-1] = 1*Sigmaz
            
            grids.append(grid)
                            
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)


def SymmetricExpSum(n,x,*args):
    #calculates f(x) = sum_i a_i*(b_i^x + b_i^{n-x})
    # args = [a_0, b_0, a_1, b_1, ...]

    if len(args)%2 == 1:
        print("Error: wrong number of args")
    k = int(len(args)/2)
    
    result = 0
    for i in range(k):
        a = args[2*i]
        b = args[2*i+1]
        result += a*(b**x + b**(n-x))
        
    return result
    
    
def SymmetricFit(fit_range,n_exp,exponent_alpha,n):
# fits 1/min{x,n-x}^\alpha with sum_i a_i*(b_i^x + b_i^{n-x}) according to the method in the Appendix of the paper

    def AdjustedFunction(xdata):
        f = np.zeros(fit_range)
        for j in range(fit_range):
            if xdata[j] <= n/2:
                f[j] = (1 / xdata[j])**exponent_alpha - 0.5 /((n-xdata[j])**exponent_alpha)
            else:
                f[j] = 0.5/(xdata[j]**exponent_alpha)
                
        return f
            
    lam, pref = fit_with_sum_of_exp(AdjustedFunction, n_exp, fit_range)

    results = [None]*(2*n_exp)
    results[::2] = pref
    results[1::2] = lam
    
    return results


class RandomIsingChainFromGrid(MPOModel):

    def __init__(self, model_params):
        n = model_params.get('L',8)
        J = model_params.get('JParams',0.5+np.zeros(n))
        h = model_params.get('hParams',0.5+np.zeros(n))
        exponent_alpha = model_params.get('exp_alpha',3)
        periodic = model_params.get('periodic',False)
        
        site = SpinHalfSite(conserve = None, sort_charge = None)
        lat = Chain(n, site, bc_MPS="finite", bc="open")
        
        Sigmax, Sigmaz, Id = site.Sigmax, site.Sigmaz, site.Id
        
        
        # define the function describing decay of interactions
        def decay(x):
            d = np.zeros(len(x))
            if periodic:
                for j in range(len(x)):
                    d[j] = (1 / min(x[j],n-x[j]))**exponent_alpha
            else:
                d = (1 / x)**exponent_alpha
            return d
        
        n_exp = 0
        
        if periodic:
            tolerance = 1e-6
        else:
            tolerance = 1e-8

        fit_range = n+n%2 # must be even
        MaxError = 1
        MaxErrorLast = 10
        MaxN = 30 # maximal number of exponentials
        
        doublecheck = 1
        check = 0
        
        # fit the decay function with iteratively more exponentials
        while MaxError > tolerance and n_exp < MaxN and doublecheck:
        
            n_exp += 1
            
            if periodic:
                
                xdata = np.arange(1,n)
                ydata = decay(xdata)

                popt = SymmetricFit(fit_range,n_exp,exponent_alpha,n)
                yfit = SymmetricExpSum(n,xdata,*popt)
                

                MaxErrorLast = MaxError
                MaxError = max(abs(ydata-yfit))
                
                if check:
                    doublecheck = 0
                
                if MaxError>MaxErrorLast and n_exp>10:
                    check = 1
                    n_exp -= 2
                
                lam = []
                pref = []
                                
                k = int(len(popt)/2)
                
                for i in range(k):
                    pref = np.append(pref, popt[2*i])
                    pref = np.append(pref, popt[2*i]*(popt[2*i+1]**n))
                    lam = np.append(lam,popt[2*i+1])
                    lam = np.append(lam,1/(popt[2*i+1]))

            else:
        
                lam, pref = fit_with_sum_of_exp(decay, n_exp, fit_range)
                
                x = np.arange(1, fit_range + 1)
                
                y1 = decay(x)
                y2 = sum_of_exp(lam, pref, x)
                
                y = np.abs(y1-y2)
                
                MaxError = y.max()
                   
        n_exp = len(lam)
                
        # this sets up the matrices of the MPO
        grids = []
        
        for i in range(n):
           
            grid = [[None for _ in range(2*n_exp+2)] for _ in range(2*n_exp+2)]
            grid[0][0] = 1*Id
            grid[0][2*n_exp+1] = h[i]*Sigmax
            grid[2*n_exp+1][2*n_exp+1] = 1*Id
           
            for j in range(n_exp):
                grid[0][j+1] = lam[j]*J[i]*Sigmaz
                grid[0][n_exp+j+1] = lam[j]*Sigmaz
                
                grid[j+1][j+1] = lam[j]*Id
                grid[n_exp+j+1][n_exp+j+1] = lam[j]*Id
                
                grid[j+1][2*n_exp+1] = pref[j]*J[i]*Sigmaz
                grid[n_exp+j+1][2*n_exp+1] = pref[j]*Sigmaz
           
            grids.append(grid)
                
        H = MPO.from_grids(lat.mps_sites(), grids, bc='finite', IdL=0, IdR=-1)
        MPOModel.__init__(self, lat, H)
