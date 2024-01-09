import numpy as np
import math
import matplotlib.pyplot as plt
import time
import cmath


from sklearn.linear_model import LassoCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


def ZOfQubitHeisenberg(periodic,n,J,a,distance):
    # general (not observable specific) neighbourhood

    MaxZLength = 2*distance+1
    
    Z = np.zeros(MaxZLength)
   
    if periodic:
    
        for i in range(distance+1):
            Z[distance+i]=J[(a+i)%n]
            Z[distance-i]=J[(a-i)%n]
    else:
       
        for i in range(distance+1):
            if a+i < n-1:
                Z[distance+i]=J[a+i]
            if a-i >= 0:
                Z[distance-i]=J[a-i]
                
    return Z
    
def PhiOfZHeisenberg(distance,R,gamma,Z,omegas):
    #maps to randomised Fourier features

    MaxZLength = 2*distance+1
    l = MaxZLength
            
    phi = np.zeros(2*R)
        
    for s in range(R):
        
        prod = np.dot(Z,omegas[s])
                                    
        phi[2*s] = math.cos(gamma/math.sqrt(l) * prod)
        phi[2*s + 1] = math.sin(gamma/math.sqrt(l) * prod)

    return phi


def CapitalPhiHeisenberg(periodic,n,distance,R,gamma,J,omegas):
    # concatenates all local feature vectors

    CPhi = []

    for i in range(n-1+int(periodic)):
    
        Z = ZOfQubitHeisenberg(periodic,n,J,i,distance)
        CPhi = np.append(CPhi,PhiOfZHeisenberg(distance,R,gamma,Z,omegas))
            
    return CPhi
    
    
class FeatureMappedLassoHeisenberg(BaseEstimator, ClassifierMixin):
    # this applies the feature mapping and then uses cross-validated Lasso model

    def __init__(self, periodic, n, distance, alphas, Omegas, R=20, Gamma=0.5):
        
        self.alphas = alphas
        self.Omegas = Omegas
        self.R = R
        self.Gamma = Gamma
        self.n = n
        self.distance = distance
        self.periodic = periodic

    def fit(self, Js, Cs):
    
        Js, Cs = check_X_y(Js,Cs)
            
        labels = LabelEncoder()
        labels.fit(Cs)
            
        self.classes_ = labels.classes_
    
        self.model = LassoCV(alphas = self.alphas)

        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        n = self.n
        distance = self.distance
        periodic = self.periodic
        
        N_train = len(Cs)
        
        results = Cs.copy()
            
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            results = np.append(results,[results,results,results,results])
        if N_train == 2:
            results = np.append(results,[results,results])
        if N_train == 3 or N_train == 4:
            results = np.append(results,results)
            
            
        PhiMatrix = np.zeros((N_train,2*R*(n-1+int(periodic))))

        for repeats in range(N_train):

            J = Js[repeats].copy()

            PhiMatrix[repeats] = CapitalPhiHeisenberg(periodic,n,distance,R,gamma,J,omegas)
            
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 2:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 3 or N_train == 4:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix))
        
        self.model.fit(PhiMatrix,results)
        return self

    def predict(self, J):

        # Check if fit has been called
        check_is_fitted(self.model)
        
        J = np.asarray(J)
        
        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        n = self.n
        distance = self.distance
        periodic = self.periodic
        
        
        if np.shape(J) != (n-1+int(periodic),):
        
            a,b = np.shape(J)
                        
            predictions = np.zeros(a)
                    
            for i in range(a):
            
                JSingle = J[i].copy()
        
                predictions[i] = self.model.predict(CapitalPhiHeisenberg(periodic,n,distance,R,gamma,JSingle,omegas).reshape(1,-1)).item()
            
        
        else:

            predictions = self.model.predict(CapitalPhiHeisenberg(periodic,n,distance,R,gamma,J,omegas).reshape(1,-1))
       
        return predictions
    
    def set_params(self, R, Gamma):
    
        self.R = R
        self.Gamma = Gamma
        
        return self
        
    def get_params(self, deep=True):
    
        return {'alphas': self.alphas, 'Omegas': self.Omegas, 'R': self.R, 'Gamma': self.Gamma, 'n': self.n, 'distance': self.distance, 'periodic': self.periodic}


def ZOfQubitIsing(periodic,n,J,h,a,distance):
    # general (not observable specific) neighbourhood

    MaxZLength = 2*(2*distance+1)
    
    Z = np.zeros(MaxZLength)
   
    if  periodic:
    
        for i in range(distance+1):
            Z[2*distance+2*i]=J[(a+i)%n]
            Z[2*distance+2*i+1]=h[(a+i)%n]
            Z[2*distance-2*i]=J[(a-i)%n]
            Z[2*distance-2*i+1]=h[(a-i)%n]
    else:
    
        for i in range(distance+1):
            if a+i < n:
                Z[2*distance+2*i]=J[a+i]
                Z[2*distance+2*i+1]=h[a+i]
            if a-i >= 0:
                Z[2*distance-2*i]=J[a-i]
                Z[2*distance-2*i+1]=h[a-i]
                
    return Z
    
def PhiOfZIsing(distance,R,gamma,Z,omegas):
    #maps to randomised Fourier features

    MaxZLength = 2*(2*distance+1)
    l = MaxZLength
            
    phi = np.zeros(2*R)
        
    for s in range(R):
        
        prod = np.dot(Z,omegas[s])
                                    
        phi[2*s] = math.cos(gamma/math.sqrt(l) * prod)
        phi[2*s + 1] = math.sin(gamma/math.sqrt(l) * prod)

    return phi


def CapitalPhiIsing(periodic,n,distance,R,gamma,J,h,omegas):
    # concatenates all local feature vectors

    CPhi = []

    for i in range(n):
    
        Z = ZOfQubitIsing(periodic,n,J,h,i,distance)
        CPhi = np.append(CPhi,PhiOfZIsing(distance,R,gamma,Z,omegas))
            
    return CPhi
    
    
class FeatureMappedLassoIsing(BaseEstimator, ClassifierMixin):
    # this applies the feature mapping and then uses cross-validated Lasso model

    def __init__(self, periodic, n, distance, alphas, Omegas, R=20, Gamma=0.5):
        
        self.alphas = alphas
        self.Omegas = Omegas
        self.R = R
        self.Gamma = Gamma
        self.n = n
        self.distance = distance
        self.periodic = periodic

    def fit(self, Js, Cs):
    
        Js, Cs = check_X_y(Js,Cs)
            
        labels = LabelEncoder()
        labels.fit(Cs)
            
        self.classes_ = labels.classes_
    
        self.model = LassoCV(alphas = self.alphas)

        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        n = self.n
        distance = self.distance
        periodic = self.periodic
        
        N_train = len(Cs)
        
        results = Cs.copy()
            
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            results = np.append(results,[results,results,results,results])
        if N_train == 2:
            results = np.append(results,[results,results])
        if N_train == 3 or N_train == 4:
            results = np.append(results,results)
            
            
        PhiMatrix = np.zeros((N_train,2*R*n))

        for repeats in range(N_train):

            JappH = Js[repeats].copy()
            J = JappH[:len(JappH)//2].copy()
            h = JappH[len(JappH)//2:].copy()
            
            PhiMatrix[repeats] = CapitalPhiIsing(periodic,n,distance,R,gamma,J,h,omegas)
            
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 2:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix,PhiMatrix))
        if N_train == 3 or N_train == 4:
            PhiMatrix = np.vstack((PhiMatrix,PhiMatrix))
        
        self.model.fit(PhiMatrix,results)
        return self

    def predict(self, J):

        # Check if fit has been called
        check_is_fitted(self.model)
        
        J = np.asarray(J)
        
        R = self.R
        gamma = self.Gamma
        omegas = self.Omegas
        n = self.n
        distance = self.distance
        periodic = self.periodic
        
        
        if np.shape(J) != (2*n,):
        
            a,b = np.shape(J)
                        
            predictions = np.zeros(a)
                    
            for i in range(a):
            
                JSingle = J[i].copy()
                JOne = JSingle[:len(JSingle)//2].copy()
                hOne = JSingle[len(JSingle)//2:].copy()
                
                predictions[i] = self.model.predict(CapitalPhiIsing(periodic,n,distance,R,gamma,JOne,hOne,omegas).reshape(1,-1)).item()
            
        
        else:
        
            JOne = J[:len(J)//2].copy()
            hOne = J[len(J)//2:].copy()

            predictions = self.model.predict(CapitalPhiIsing(periodic,n,distance,R,gamma,JOne,hOne,omegas).reshape(1,-1))
       
        return predictions
    
    def set_params(self, R, Gamma):
    
        self.R = R
        self.Gamma = Gamma
        
        return self
        
    def get_params(self, deep=True):
    
        return {'alphas': self.alphas, 'Omegas': self.Omegas, 'R': self.R, 'Gamma': self.Gamma, 'n': self.n, 'distance': self.distance, 'periodic': self.periodic}
