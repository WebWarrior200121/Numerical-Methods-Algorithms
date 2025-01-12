import numpy as np

# integration using extended trapezoidal rule
def trapz_n(f, a, b, s, n):
    ''' 
    trapz_n: integrates f from a to b using the trapezoidal
    rule with N = 2**(n - 1) intervals.
    * To compute the integral with some n, it is required that 
      the integral has been computed for previous (n - 1) and the 
      corresponding value of s been saved to be used as input.      
    * This means that s is a input and output variable. 
    * Then trapz_n must be called with succesive values n = 1, 2, ... 
      saving the each corresponding s for next calculation.    
    '''
    if (n == 1):
        # just trapezoidal with one interval
        s = 0.5*(b - a)*(f(a) + f(b))
    else:
        # adjust new step
        new = 2**(n - 2)   # number of new points
        h = (b - a)/new    # step in previous iteration
        
        # compute the contribution of new points 
        sum = 0.
        for j in range(1, new + 1):
            x = a + (j - 0.5)*h
            sum = sum + f(x)

        # update new integral
        s = 0.5*(s + h*sum)

    return s


# integration using an iterative trapezoidal algorithm
def trapz(f, a, b, eps=10**(-10)):

    nmax = 24
    zero = 10.**(-15)
    
    h = (b - a)
    sold = 0.
    snew = 0.5*h*(f(a) + f(b))
    
    for n in range(2,nmax+1):
        
        term = 0.
        sold = snew
        
        # compute the contribution of new points
        for k in range(1, 2**(n - 2) + 1):
            term = term + f(a + (k - 0.5)*h)
        
        snew = 0.5*(sold + h*term)
        h = 0.5*h  # step for next iteration
        
        # print("n: %2d, integral: %.16f, relative error: %e"%(n, snew, abs(snew-sold)/abs(snew)))
        if (n > 6):
            if( abs(snew-sold) < abs(snew)*eps or (abs(snew) <= zero and abs(sold) <= zero) ):
                return (n, snew)

    # To return a value anyway we give the value in the
    # last iteration.
    print('trapz: nmax reached without convergence, result could be inaccurate.')
    return (n, snew)


### RECURSIVE TRAPEZOID INTEGRATION ALGORITHM USING PYTHON RECURSION

def trap_iter(f, a, b, n):
    ''' 
    trapz_n: integrates f from a to b using the trapezoidal
    rule with N = 2**(n - 1) intervals.
    * To compute the integral with some n, it is required that 
      the integral has been computed for previous (n - 1) and the 
      corresponding value of s been saved to be used as input.      
    * This means that s is a input and output variable. 
    * Then trapz_n must be called with succesive values n = 1, 2, ... 
      saving the each corresponding s for next calculation.    
    '''

    if (n == 1):
        # just trapezoidal with one interval
        s = 0.5*(b - a)*(f(a) + f(b))
        return s
    else:
        s = trap_iter(f,a,b,n-1)
        #print(s)
        # adjust new step
        new = 2**(n - 2)   # number of new points
        h = (b - a)/new    # step in previous iteration

    
        # compute the contribution of new points 
        sum = 0.
        for j in range(1, new + 1):
            x = a + (j - 0.5)*h
            sum = sum + f(x)
        # update new integral
        s = 0.5*(s + h*sum)

    return s



# NEVILLE - ROMBERG INTEGRATION

import numerics as nm

def int_rich(f, a,b, M=8, debug = False):
    # M is max iterations of the richardson method
    
    r = 2
    
    # Initialize the differences matrix
    D = np.zeros((M, M), dtype = np.float64)
    E = np.zeros((M, M), dtype = np.float64)
    
    
    
    #Initialize matrix element D[0,0]
    
    # In this case the number of subinterval is N = 2**(n-1). n =1 is one subinterval
    n = 1
    s= 0
    D[0,0] = trapz_n(f, a, b, s, n)
    
    E[0,0] = 10 ** 30
    error_max = E[0,0]
    drow = D[0,0]
    drowold = drow
    error_max_old = error_max
    
    
    for i in range(1,M):
        #In this case, make h = h / 2 means, n = n+1. 
        n = n+1
        D[i,0] = trapz_n(f, a, b, D[i-1,0], n)
        E[i,0] = abs(D[i,0]-D[i-1,0])
        
        
        
        for j in range(1,i+1):
    
            D[i,j] = D[i-1,j-1] - ((r**2)**j)*D[i,j-1]
            D[i,j] = D[i,j] / (1-((r**2)**j))
            E[i,j] = max(abs(D[i,j]-D[i-1,j-1]),abs(D[i,j]-D[i,j-1]))
            error = E[i,j]

            if error < error_max:
                error_max_old = error_max
                error_max = error
                drowold = drow
                drow = D[i,j]
        

        if (abs(drow - drowold) >= 2.0 * error_max or abs(D[i,i] - D[i-1,i-1]) >= 2.0 * error_max): 
            if debug:
                print("Iterations reached: ", i)
                print(D)
            return drowold, error_max
            

    if debug:
        print(D)
        print("No exit condition. Max iterations reached: ", i)

    return drow,error_max


def romberg_rich(f, a,b, M=8, tol = 10**(-6)):
    # M is max iterations of the richardson method
    
    r = 2
    
    # Initialize the differences matrix
    D = np.zeros((M, M), dtype = np.float64)
    E = np.zeros((M, M), dtype = np.float64)
    
    
    
    #Initialize matrix element D[0,0]
    
    # In this case the number of subinterval is N = 2**(n-1). n =1 is one subinterval
    n = 1
    s= 0
    D[0,0] = trapz_n(f, a, b, s, n)
    
    E[0,0] = 10 ** 30
    error_max = E[0,0]
    drow = D[0,0]
    drowold = drow
    error_max_old = error_max
    
    
    for i in range(1,M):
        #In this case, make h = h / 2 means, n = n+1. 
        n = n+1
        D[i,0] = trapz_n(f, a, b, D[i-1,0], n)
        E[i,0] = abs(D[i,0]-D[i-1,0])
        
        
        
        for j in range(1,i+1):
    
            D[i,j] = D[i-1,j-1] - ((r**2)**j)*D[i,j-1]
            D[i,j] = D[i,j] / (1-((r**2)**j))
            E[i,j] = max(abs(D[i,j]-D[i-1,j-1]),abs(D[i,j]-D[i,j-1]))
            error = E[i,j]

            if error < error_max:
                error_max_old = error_max
                error_max = error
                drowold = drow
                drow = D[i,j]
        
        if ( abs(D[i,i] - D[i-1,i-1]) <= tol * D[i,i] + tol): 
            return drowold
            


    print("No exit condition. Max iterations reached: ", i)

    return drow


# ROMBERG NEVILLE 

def romberg_neville(f,a,b,tol =10**(-10)):
    m = 5 #m is the number of points for interpolation. m-1 is the degree of the interpolation polynomial
    n = 10 #n is the max number of iterations
    
    # Arrays definitions.
    Z = np.zeros(m, dtype=np.float64) # We are making the substitution z = h^2.
    S  = np.zeros(m, dtype=np.float64) #Trap Sum/Integral data
    
    
    #h = (b-a) # initial step
    delta = 1 #delta = h/(b-a) #Re-scale
    z = delta**2  # initial z
    
    #ITERATIVE COMPOSITE TRAPEZOIDAL INTEGRATION
    
    N = 1 # 2^(N-1) is the number of subintervals. N = 1 is the initial value
    s_old=0 # s  is the PREVIOUS trapezoidal summation. 
    S[0] = nm.trapz_n(f, a, b, s_old, N)
    Z[0] = z
    
    #Initial set of data points
    for i in range(1,m):
        N = N+1
        s_old = S[i-1]
        delta = delta/2
        z = (delta**2)
        S[i] = nm.trapz_n(f, a, b, s_old, N)
        Z[i] = z

    
    for j in range(n):
        integral, error = nm.neville_optimized(Z,S, 0) #Extrapolation at z = 0
    
        if error < tol * abs(integral) + tol:
            return integral, error
        else:
            
            # REORGANIZE DATA
            for k in range(0,m-1):
                S[k] = S[k+1]
                Z[k] = Z[k+1]
            
            # We have now s[m-1] free
            N = N+1
            s_old = S[m-2]
            delta = delta/2
            z = (delta**2)
            
            S[m-1] = nm.trapz_n(f, a, b, s_old, N)
            Z[m-1] = z
            if (np.abs(Z[m-1]-Z[m-2]) == 0):
                print("Step h cant be decreased")
                return integral, error
        
        
    
    return integral,error


#GAUSSIAN QUADRATURES

def gauleg_rw(N):
    # the P_N Legendre polynomial has N roots
    
    root_arr = np.zeros(N, dtype=np.float64)-1
    weight_arr = np.zeros(N, dtype=np.float64)
    
    
    f = lambda x: nm.legendre_poly(x, N)
    df = lambda x: nm.der_legendre_poly(x,N)
    
    # M is the number of searches we need. Because of the symmetries of 
    # the polynomials M < N

    
    # If N is even, we only need to find N / 2 zeros
    
    N_odd = False
    
    if N % 2 == 0:
        M = N // 2
    
    else:
        # Because of the odd behaviour of odd hermite polynomials, dont need to find one root.
        # The zero is always a root for odd polynomials
        M = (N-1)//2
        N_odd = True
        
        #The position of the middle term is M+1. The middle root is zero
        root_arr[(M+1-1)]= 0
        weight_arr[(M+1-1)] = 2 / (df(0))**2
        


    
    
    # i is the i-th positive root. i=1,2,3,...M

    for i in range(1,M+1):

        # Initial guess for the i-th root
        initial_guess_i = (4 * i - 1)/(4 * N + 2)
        initial_guess_i = np.cos(np.pi * initial_guess_i)
        
        # Newton method to find the i-th root
        root,derr = nm.newton_gq(f, df, initial_guess_i)
        
    
        weight = (1-root ** 2) * ((derr) ** 2)
        weight = 2 / weight
        
        
        if N_odd:
            # The middle element is M+1, the upper i element is M+1+i. The indexation is M+1+i-1
            root_arr[((M+1)+i)-1] = -root
            weight_arr[((M+1)+i)-1] = weight

            # The middle element is M+1, the lower i element is M+1-i. The indexation is M+1-i-1
            root_arr[((M+1)-i)-1] = root
            weight_arr[((M+1)-i)-1] = weight


        else:

            root_arr[(M+i)-1] = -root
            weight_arr[(M+i)-1] = weight

            root_arr[(M-i+1)-1] = root
            weight_arr[(M-i+1)-1] = weight
        
    
    return root_arr,weight_arr


def gauleg_rw_2(N):
    # the P_N Legendre polynomial has N roots
    
    root_arr = np.zeros(N, dtype=np.float64)-1
    weight_arr = np.zeros(N, dtype=np.float64)
    
    if N > 2:
        f = lambda x: nm.legendre_poly(x, N)
        df = lambda x: nm.der_legendre_poly(x,N)
    
    elif N ==1:
        f = lambda x: 1
        df = lambda x: 0
    
    elif N==2:
        f = lambda x: 2 * x
        df = lambda x: 2
    
    # M is the number of searches we need. Because of the symmetries of 
    # the polynomials M < N

    
    # If N is even, we only need to find N / 2 zeros
    
    N_odd = False
    
    if N % 2 == 0:
        M = N // 2
    
    else:
        # Because of the odd behaviour of odd hermite polynomials, dont need to find one root.
        # The zero is always a root for odd polynomials
        M = (N-1)//2
        N_odd = True
        
        #The position of the middle term is M+1. The middle root is zero
        root_arr[(M+1-1)]= 0
        weight_arr[(M+1-1)] = 2 / (df(0))**2
        


    
    
    # i is the i-th positive root. i=1,2,3,...M

    for i in range(1,M+1):

        # Initial guess for the i-th root
        initial_guess_i = (4 * i - 1)/(4 * N + 2)
        initial_guess_i = np.cos(np.pi * initial_guess_i)
        
        # Newton method to find the i-th root
        root,derr = nm.newton_gq(f, df, initial_guess_i)
        
    
        weight = (1-root ** 2) * ((derr) ** 2)
        weight = 2 / weight
        
        
        if N_odd:
            # The middle element is M+1, the upper i element is M+1+i. The indexation is M+1+i-1
            root_arr[((M+1)+i)-1] = -root
            weight_arr[((M+1)+i)-1] = weight

            # The middle element is M+1, the lower i element is M+1-i. The indexation is M+1-i-1
            root_arr[((M+1)-i)-1] = root
            weight_arr[((M+1)-i)-1] = weight


        else:

            root_arr[(M+i)-1] = -root
            weight_arr[(M+i)-1] = weight

            root_arr[(M-i+1)-1] = root
            weight_arr[(M-i+1)-1] = weight
        
    
    return root_arr,weight_arr

def gauleg_integral(f,a,b,N):
    root_arr, weight_arr = gauleg_rw(N)
    points_eval = ((b-a)/2)* root_arr + (b+a)/2

    integral = 0
    
    for i in range(N):
        integral = integral + weight_arr[i] * f(points_eval[i])
    integral = ((b-a)/2)* integral
    return integral

def gauleg_integral_2(f,a,b,N):
    root_arr, weight_arr = gauleg_rw_2(N)
    points_eval = ((b-a)/2)* root_arr + (b+a)/2

    integral = 0
    
    for i in range(N):
        integral = integral + weight_arr[i] * f(points_eval[i])
    integral = ((b-a)/2)* integral
    return integral


import numerics as nm
import numpy as np
def pw_gq(f,a,b,N,m):
    # N number of intervals (not of points of the partition)
    # There are N+1 partition points
    # m numbers of quadrature points (per interval integration)
    
    h = (b-a)/(N)
    
    # Definition of the partition (points)
    X = np.array([a + i * h for i in range(0,N+1)], dtype = np.float64)
    
    S = 0
    
    for i in range(N):
        S += nm.gauleg_integral(f,X[i],X[i+1],m)
    
    return S



def trapezoidal_composite(f, a, b, s, n, return_count = False, old_count = 0):
    ''' 
    trapz_n: integrates f from a to b using the trapezoidal
    rule with N = 2**(n - 1) intervals.
    * To compute the integral with some n, it is required that 
      the integral has been computed for previous (n - 1) and the 
      corresponding value of s been saved to be used as input.      
    * This means that s is a input and output variable. 
    * Then trapz_n must be called with succesive values n = 1, 2, ... 
      saving the each corresponding s for next calculation.    
    '''
    
    if (n == 1):

        # just trapezoidal with one interval
        s = 0.5*(b - a)*(f(a) + f(b))
        if return_count:
            return s,2
    else:
        # adjust new step
        count = old_count
        new = 2**(n - 2)   # number of new points
        h = (b - a)/new    # step in previous iteration
        # compute the contribution of new points 
        sum = 0.
        for j in range(1, new + 1):
            x = a + (j - 0.5)*h
            sum = sum + f(x)
            count += 1
        # update new integral
        s = 0.5*(s + h*sum)
        
        if return_count:
            return s,count

    return s

def simpson_composite(f, a, b, n, return_count = False):
    # Number of subintervals N = 2**(n-1)
    s = 0 # s is temp variable.
    count = 0 
    for i in range(1,n+1):
        if return_count:
            s,count = trapezoidal_composite(f, a, b, s, i, return_count = True, old_count = count)
        else:
            s = trapezoidal_composite(f, a, b, s, i)
        if i == n-1:
            s0 = s # when i = n-1, s -> trapezoidal for N/2 subintervals
        
    # when i = n, s-> trapezoidal for N subintervals
    
    if return_count:
        return (4.0/3) * s - (1.0/3) * s0,count
    
    return (4.0/3) * s - (1.0/3) * s0


def Gauss(f,a,b):
    return nm.gauleg_integral(f,a,b,2)
def Adaptive(f,sum,a,b,depth):
    tol = 10**(-14)
    max = 12
    #print("[a,b]",a,b)
    if depth > max:
        print("Error")
        return None
    else:
        c = (a+b)/2
        left = Gauss(f,a,c)
        right = Gauss(f,c,b)
        int1 = Gauss(f,a,b)
        int2 = left + right
        #print(int1,int2)
        #print(abs(int2-int1)/abs(int2), "depth", depth)
        if abs(int2-int1) <= tol * abs(int2) + tol:
            sum = sum + int2
            return sum
        else:
            return Adaptive(f,sum,a,c,depth+1)+Adaptive(f,sum,c,b,depth+1)
    
def AdaptiveInitial(f,a,b):
    sum = 0
    depth = 0
    sum = Adaptive(f,sum,a,b,depth)
    return sum