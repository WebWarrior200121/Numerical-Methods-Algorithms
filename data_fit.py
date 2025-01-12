import numerics as nm
import numpy as np
from scipy import linalg
from scipy.special import gammainc

def general_fit(X,Y,SIGMA,M):
    N = len(X) # The data set have N points
    
    # M is the number of parameters a_0,a_1,...,a_{M-1}: a0 + a_1 x^1 + ... + a_{M-1} x^{M-1}
    ALPHA = np.zeros((M,M), dtype = np.float64)
    BETA = np.zeros(M, dtype = np.float64)
    
    
    for k in range(M):
        for j in range(M):
            ALPHA[k,j] = np.sum([ ((X[i])**j * X[i]** k)/(SIGMA[i]**2) for i in range(N)])
        
        
        BETA[k] = np.sum([ (Y[i] * ((X[i])**k) ) / ((SIGMA[i]) ** 2) for i in range(N)])
        
    ALPHA_LU, P = nm.LU(ALPHA, return_lu_matrices = False, determinant_calculation = False)
    
    # CALCULATION OF THE COEFFICIENTS a0,a1,a2...
    
    A = nm.LUsolve(ALPHA_LU,BETA,return_lu_matrices = False, already_factorized = True, P_given = P)

    
    #CALCULATION OF THE INVERSE OF ALPHA
    
    C = np.zeros((M,M), dtype = np.float64)
    for k in range(M):
        I = np.zeros(M, dtype = np.float64)
        I[k] = 1
        C[:,k] = nm.LUsolve(ALPHA_LU,I,return_lu_matrices = False,already_factorized = True,P_given = P)
    
    SIGMA_A = np.array([np.sqrt(C[j,j]) for j in range(M)])
    
    T = np.zeros(N,dtype = np.float64)
    for i in range(N):
        T[i] = np.sum([A[k] * (X[i])**k for k in range(M)])
        
    chi_square = np.sum([ ( (Y[i] - T[i])/(SIGMA[i]) )**2 for i in range(N)])
    
    degrees_of_freedom = N - M  # Number of data points minus the number of estimated parameters

    # Calculation of quality factor using gamma and incomplete gamma functions
    Q = 1 - gammainc(degrees_of_freedom / 2, chi_square / 2)
    
    if (Q < 0.05):
        print("Reject model")
    
    
    return A,SIGMA_A, chi_square, Q, condition_number(ALPHA)




def condition_number(A):
    return np.linalg.cond(A, 'fro')
    
