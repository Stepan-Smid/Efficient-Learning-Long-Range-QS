import numpy as np
import random as rnd
import math
import cmath

import scipy
from scipy import sparse as sps

import tenpy
from tenpy.networks.mpo import MPO
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

from models import HeisenbergChainOpenBC, HeisenbergChainFromGrid, RandomIsingChainFromGrid


XGate = [[0,1],[1,0]]
YGate = [[0,0-1j],[0+1j,0]]
ZGate = [[1,0],[0,-1]]

    
def CalculatePropertiesDMRGHeisenberg(periodic,n,J):
    # calculates the required property using DMRG
    
    model_params = dict(L=n, JParams = J)
    
    if periodic:
        Heisenberg = HeisenbergChainFromGrid(model_params)

    else:
        Heisenberg = HeisenbergChainOpenBC(model_params)
        
    # sets up initial MPS as a random product state
    product_state = []
    for i in range(Heisenberg.lat.N_sites):
        product_state.append(rnd.choice(["up", "down"]))
    psi = MPS.from_product_state(Heisenberg.lat.mps_sites(), product_state, bc = Heisenberg.lat.bc_MPS)

    dmrg_params = {
        'mixer': False,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-6,
        'trunc_params': {
            #'chi_max': 10,
            'svd_min': 1.e-9
        },
        'combine': True,
        'chi_list': {0: 5, 5: 10, 15: 20, 25: 50, 90:100},
        'max_sweeps': 100

    }
    
    Heisenberg.test_sanity()
        
    info = dmrg.run(psi, Heisenberg, dmrg_params)
        
    """ZZ = npc.outer(psi.sites[0].Sigmaz.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmaz.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    YY = npc.outer(psi.sites[0].Sigmay.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmay.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    XX = npc.outer(psi.sites[0].Sigmax.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].Sigmax.replace_labels(['p', 'p*'], ['p1', 'p1*']))
        
    Gates = XX + YY + ZZ
    
    C = 1/3 *psi.expectation_value(Gates)"""
            
    return info['E']/math.sqrt(n)

 
def CalculatePropertiesExactHeisenberg(periodic,n,J):
    # calculates the required property using exact diagonalisation


    H = sps.csr_matrix((2**n,2**n))
    
    if periodic:
        ran = n
    else:
        ran = n-1
    
    for i in range(ran):
    
        j = (i+1)%n

        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        YiGate = sps.kron(sps.kron(sps.eye(2**i),YGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))
                
        XjGate = sps.kron(sps.kron(sps.eye(2**j),XGate),sps.eye(2**(n-j-1)))
        YjGate = sps.kron(sps.kron(sps.eye(2**j),YGate),sps.eye(2**(n-j-1)))
        ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))

        Gates = np.real((XiGate*XjGate + YiGate*YjGate + ZiGate*ZjGate))

        H += J[i] * Gates
        
    eval, evec = sps.linalg.eigsh(H,k=1,which="SA")
    
    """C = np.zeros(n-1)
    #C = (C_01, C_12, ... , C_{n-2,n-1})

    for i in range(ran):
    
        j = (i+1)%n

        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        YiGate = sps.kron(sps.kron(sps.eye(2**i),YGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))
                
        XjGate = sps.kron(sps.kron(sps.eye(2**j),XGate),sps.eye(2**(n-j-1)))
        YjGate = sps.kron(sps.kron(sps.eye(2**j),YGate),sps.eye(2**(n-j-1)))
        ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))

        Gates = 1/3 * (XiGate*XjGate + YiGate*YjGate + ZiGate*ZjGate)
            
        C[i] = np.real((np.conj(np.transpose(evec)).dot(Gates.dot(evec))).item())"""
    
    return eval.item()/math.sqrt(n)


def CalculatePropertiesDMRGIsing(periodic,n,J,h,exponent_alpha):
    # calculates the required property using DMRG

    C = np.zeros(n)
    
    model_params = dict(L=n, bc="open", bc_MPS="finite", JParams = J, hParams = h, exp_alpha = exponent_alpha, periodic = periodic)

    Ising = RandomIsingChainFromGrid(model_params)
        
    product_state = []
    for i in range(Ising.lat.N_sites):
        product_state.append(rnd.choice(["up", "down"]))
    psi = MPS.from_product_state(Ising.lat.mps_sites(), product_state, bc = Ising.lat.bc_MPS)

    dmrg_params = {
        'mixer': False,  # setting this to True helps to escape local minima
        'max_E_err': 1.e-6,
        'trunc_params': {
            #'chi_max': 10,
            'svd_min': 1.e-9
        },
        'combine': True,
        'chi_list': {0: 5, 5: 10, 15: 20, 25: 50, 90: 100},
        'max_sweeps': 100
    }
    
    Ising.test_sanity()
        
    info = dmrg.run(psi, Ising, dmrg_params)
        
    return info['E']/math.sqrt(n)


    
def CalculatePropertiesExactIsing(periodic,n,J,h,exponent_alpha):
    # calculates the required property using exact diagonalisation

    H = sps.csr_matrix((2**n,2**n))
    
    def dist(i,j):
        if periodic:
            d = min(abs(i-j), n-abs(i-j))
        else:
            d = abs(i-j)
        return d
    
    for i in range(n):
    
        XiGate = sps.kron(sps.kron(sps.eye(2**i),XGate),sps.eye(2**(n-i-1)))
        ZiGate = sps.kron(sps.kron(sps.eye(2**i),ZGate),sps.eye(2**(n-i-1)))

        H += h[i] * XiGate
    
        for j in range(n):
        
            if i < j:
                                    
                ZjGate = sps.kron(sps.kron(sps.eye(2**j),ZGate),sps.eye(2**(n-j-1)))
                
                H += (1+J[i]*J[j]) / (dist(i,j)**exponent_alpha) * ZiGate * ZjGate
                                
    eval, evec = sps.linalg.eigsh(H,k=1,which="SA")
    
    return eval.item()/math.sqrt(n)
