import numpy as np
import random as rnd
import math
import matplotlib.pyplot as plt
import time
import cmath

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from lasso import FeatureMappedLassoHeisenberg, FeatureMappedLassoIsing

from observables import CalculatePropertiesExactHeisenberg, CalculatePropertiesDMRGHeisenberg, CalculatePropertiesExactIsing, CalculatePropertiesDMRGIsing


n = 32 # number of qubits

# parameters of the Hamiltonians
h_max = math.e
exponent_alpha = 3
J_max = 2


UsedModel = 1 # 0 = Heisenberg, 1 = Ising
periodic = False # boundary conditions open/periodic, use with care
# Note that when True, this won't actually show O(1) scaling, as the predicted observable is still the energy, and the feature mapping used is a 'general' one. To see O(1) scaling, one needs to predict only a p-body observable and use only its local feature vector, and then use this model to predict all such p-body observables to calculate their sum
# When using periodic Ising at a given size/parameters for the first time, be mindful of how many exponentials get used for the fit, might need to play with the tolerances etc.


# hyperparameters of the ML model
alphas = [2**(-8), 2**(-7), 2**(-6), 2**(-5)]
Rs = [5, 10, 20, 40]
gammas = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75]


ErrorTol = 0.3 # determines the additive error epsilon, set mindfully


N_test = 40 # the number of testing samples
N_train = 0 # initial number of training samples; this should principally be 0, increase only if you need to speed things up and you have an idea of how many training data you will need

distance = 4 # the distance delta determining the size of neighbourhood of parameters used to approximate a given p-body observable
    
    
def InitiateChain():
# sets the random parameters in the Hamiltonians

    if UsedModel == 0: # Heisenberg chain
        
        h = None
        
        if periodic:
            # gives J = (J_01, J_12, ... , J_{n-2,n-1}, J_{n-1,0})
    
            J = J_max * np.random.rand(n)
    
            if n == 2:
                J[1] = 0
        
        else:
            # gives J = (J_01, J_12, ... , J_{n-2,n-1})
            
            J = J_max * np.random.rand(n-1)
            
    if UsedModel == 1: # Ising chain
        # gives J[i] = J_i, h[i] = h_i

        J = J_max * np.random.rand(n)
        #h = h_max * np.random.rand(n)
        h = np.zeros(n) + h_max
   
    return J,h
    
    
def CalculatePropertiesExact(J,h):
# evaluates the observable for the given parameters using exact diagonalisation
# faster for smaller chains, up to ~16 qubits, good for comparison

    if UsedModel == 0:# Heisenberg chain
        
        C = CalculatePropertiesExactHeisenberg(periodic,n,J)
           
    if UsedModel == 1:# Ising chain
       
       C = CalculatePropertiesExactIsing(periodic,n,J,h,exponent_alpha)
   
    return C
    
    
def CalculatePropertiesDMRG(J,h):
# evaluates the observable for the given parameters using DMRG

    if UsedModel == 0:# Heisenberg chain
           
        C = CalculatePropertiesDMRGHeisenberg(periodic,n,J)
           
    if UsedModel == 1:# Ising chain
       
       C = CalculatePropertiesDMRGIsing(periodic,n,J,h,exponent_alpha)
   
    return C
    

CurrentError = 10 + ErrorTol # needs to be initially > ErrorTol


# sets up the testing samples with outcomes
JTests = []
CTests = []

for i in range(N_test):

    print("Setting up test",i+1,"/",N_test)

    J,h = InitiateChain()
    
    if UsedModel == 0:
        JTests.append(J)
    if UsedModel == 1:
        JTests.append(np.append(J,h))
        
    C = CalculatePropertiesDMRG(J,h)
    CTests.append(C)


# shows the range and standard deviation of observables, useful for determining the required ErrorTol and when checking the concentration of expectation values
rangeTests = np.asarray(CTests).max()-np.asarray(CTests).min()
print("\nRange is ",rangeTests)
print("Standard deviation is ",np.std(CTests),"\n")


# sets up the initial training samples with outcomes
Js = []
Cs = []
    
for i in range(N_train):

    print("Setting up train",i+1,"/",N_train)

    J,h = InitiateChain()
    
    if UsedModel == 0:
        Js.append(J)
    if UsedModel == 1:
        Js.append(np.append(J,h))

    Cs.append(CalculatePropertiesDMRG(J,h))
    
    
while CurrentError > ErrorTol:

    # add one more training sample
    N_train += 1
 
    J,h = InitiateChain()
    
    if UsedModel == 0:
        Js.append(J)
    if UsedModel == 1:
        Js.append(np.append(J,h))
        
    Cs.append(CalculatePropertiesDMRG(J,h))
    
        
    @ignore_warnings(category=ConvergenceWarning) #ignoring convergence warning
    def AvgError():

        if UsedModel == 0:
            MaxZLength = 2*distance+1
        if UsedModel == 1:
            MaxZLength = 2*(2*distance+1)

        omegas = np.random.normal(0,1,(max(Rs),MaxZLength))
 
        param_grid = dict(R = Rs, Gamma = gammas)
        
        if UsedModel == 0:
            model = FeatureMappedLassoHeisenberg(periodic, n, distance, alphas, omegas)
        if UsedModel == 1:
            model = FeatureMappedLassoIsing(periodic, n, distance, alphas, omegas)

        
        # need to have at least 5 training samples for CV, LassoCV, etc; so just duplicate them if less than 5
        if N_train == 1:
            CsElongated = np.append(Cs,[Cs,Cs,Cs,Cs])
            JsElongated = np.vstack((Js,Js,Js,Js,Js))
        elif N_train == 2:
            CsElongated = np.append(Cs,[Cs,Cs])
            JsElongated = np.vstack((Js,Js,Js))
        elif N_train == 3 or N_train == 4:
            CsElongated = np.append(Cs,Cs)
            JsElongated = np.vstack((Js,Js))
        else:
            JsElongated = Js.copy()
            CsElongated = Cs.copy()
                    
                    
        # train the model using 5-fold cross-validation
        grid = GridSearchCV(model,param_grid,cv = 5, scoring = 'neg_root_mean_squared_error', return_train_score = False)
        grid.fit(np.asarray(JsElongated),np.asarray(CsElongated))
        
        
        """# determine the coeff of determination of the fit
        variance = 0
        mean = np.asarray(CsElongated).mean()
        k = len(CsElongated)
        for j in range(len(CsElongated)):
            variance += (CsElongated[j]-mean)**2
        score = grid.score(np.asarray(JsElongated),np.asarray(CsElongated))
        print("Coeff of determination is: ",1-k*(score**2)/variance)"""
        
        error = 0

        for repeats in range(N_test):

            J = np.asarray(JTests[repeats].copy())
            result = CTests[repeats]
                                                
            prediction = grid.predict(J).item()
            
            #print("Exact is",result," and prediction is",prediction)
                            
            error += 1/N_test *((result - prediction)**2)

        #print("Best parameters are:",grid.best_params_,"Alpha = ",grid.best_estimator_.model.alpha_)
                    
        del(grid)
                    
        return math.sqrt(error)
        
    CurrentError = AvgError()
    
    print("\nN_train is ",N_train," and the error is ",CurrentError,"\n")
