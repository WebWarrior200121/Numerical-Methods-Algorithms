import numpy as np
import numerics as nm

def gauss_solve(A, b):
    n = len(A)
    Ab = np.concatenate((A, b), axis=1)  # Augmented matrix

    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        pivot = Ab[i, i]
        if pivot == 0:
            raise ValueError("No unique solution exists.")

        Ab[i] /= pivot
        for j in range(i + 1, n):
            Ab[j] -= Ab[j, i] * Ab[i]

    # Back substitution
    x = np.zeros(n)
    x[n - 1] = Ab[n - 1, n]
    for i in range(n - 2, -1, -1):
        x[i] = Ab[i, n] - np.dot(Ab[i, i + 1:n], x[i + 1:n])

    return x



#LU FACTORIZATION
def LU(A0, return_lu_matrices = False, determinant_calculation = False):
    A = A0.copy()
    n = len(A)

    # Permutation array/vector
    P = np.array([i for i in range(n)])
    
    if return_lu_matrices:
        U = np.zeros((n,n), dtype = np.float64)
        L = np.zeros((n,n), dtype = np.float64)
        
       
    #Number of permutations
    S = 0
    # U determinant
    U_det = 1
    
    # Forward elimination with partial pivoting
    for i in range(n):
        # Partial pivoting
        max_row = np.argmax(np.abs(A[i:, i])) + i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]] #THE FACTORS ALSO NEED TO PERMUTATE
            P[[i,max_row]] = P[[max_row,i]]
            S = S + 1
            
            if return_lu_matrices:
                L[[i,max_row]] = L[[max_row,i]]          


        pivot = A[i, i]
        if pivot == 0:
            raise ValueError("No unique solution exists.")
            
        U_det = U_det * pivot


        for j in range(i + 1, n):
            factor = A[j,i]/pivot
            A[j,i:] -= factor * A[i,i:] #Remember only change the elements on and above the diagonal of the matrix. Otherwise, will we change the factors (entries of L)
            A[j,i] = factor #Replace the zero by the factor
            
            if return_lu_matrices:
                L[j,i] = factor
        
        if return_lu_matrices:
            L[i,i] = 1
            U[i,i:] = A[i,i:] 

    if determinant_calculation:
        det = (-1)**S * U_det
        return A,P,det
    
    if return_lu_matrices:
        A,P,L,U
 
    return A,P

def LUsolve(A0,b,return_lu_matrices = False, already_factorized = False, P_given = False):
    
    #TO SOLVE FOR MANY B'S WE FIRST CALL LU AND THEN CALL LUsolve WITH already_factorized = True, P_given = ...
    n = len(A0)
    
    if already_factorized:
        A = A0
        P = P_given
    else:
        A,P= LU(A0)

    #Permutation of the b in order to use backward o forward substitution
    b_permuted = np.zeros(n, dtype = np.float64)
    for i in range(n):
        b_permuted[i] = b[P[i]]

    y = forward_substitution_ones(A, b_permuted)
    x = backward_substitution(A, y)
    
    if return_lu_matrices:
        return x,L,U
    return x



def inverse_det(A0):
    A,P,S,U_det = LU(A0, return_lu_matrices = False, determinant_calculation = True)
    
    det = (-1)**S * U_det
    
    n = len(A)
    inverse = np.zeros((n,n), dtype = np.float64)
    for k in range(n):
        I = np.zeros(n)
        I[k] = 1
        inverse[:,k] = LUsolve(A,I,return_lu_matrices = False,already_factorized = True,P_given = P)
        x = inverse[:,k]
    
    return inverse,det


def inverse(A0,P):
    A = A0.copy()    
    n = len(A)
    inverse = np.zeros((n,n), dtype = np.float64)
    for k in range(n):
        I = np.zeros(n)
        I[k] = 1
        inverse[:,k] = LUsolve(A,I,return_lu_matrices = False,already_factorized = True,P_given = P)
        x = inverse[:,k]
    
    return inverse
    
    
    
def forward_substitution_ones(L0, b0):
    L = L0.copy()
    b = b0.copy()
    n = L.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        L[i,i] = 1
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

def backward_substitution(U0, b0):
    U = U0.copy()
    b = b0.copy()
    n = U.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    return x


def forward_substitution(L0, b0):
    L = L0.copy()
    b = b0.copy()
    n = L.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x



#FITTING

from scipy.special import gammainc
import numpy as np


def linear_regression(X,Y,SIGMA):
    
    #DEFINITION OF VALUES
    
    S = 0
    Sx = 0
    Sy = 0
    Sxx = 0
    Sxy = 0
    
    n = len(X)
    for i in range(n):
        term = 1 / SIGMA[i]**2
        xi = X[i]
        yi = Y[i]
        S = S + term
        
        Sx = Sx + xi * term
        Sy = Sy + yi * term
        Sxx = Sxx + (xi ** 2) * term
        Sxy = Sxy + (xi * yi) * term
    
    
    Delta = S * Sxx - (Sx)**2
    
    a = (Sxx * Sy - Sx * Sxy) / Delta
    b = (S * Sxy - Sx * Sy) / Delta
    
    sigma_a = np.sqrt(Sxx / Delta)
    sigma_b = np.sqrt(S / Delta)
    
    residuals = Y - (a + b * X)
    array = np.divide(residuals,SIGMA)
    array = array ** 2
    chi_squared = np.sum(array)
    
    degrees_of_freedom = n - 2  # Number of data points minus the number of estimated parameters

    # Calculation of quality factor using gamma and incomplete gamma functions
    Q = 1 - gammainc(degrees_of_freedom / 2, chi_squared / 2)
    
    if (Q < 0.05):
        print("Reject model")

    return a, b, sigma_a, sigma_b, chi_squared, Q



        