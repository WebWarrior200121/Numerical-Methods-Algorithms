# import modules

import numpy as np
import warnings

# Quadratic Equation Solver
# a*x**2+b*x+c 0

_w_e2solve_infsol = 'eq2solve@infsol: a=b=c=0, equation has infinite solutions'
_w_e2solve_nosol = 'eq2solve@nosol: a=b=0 and c!=0, equation has no solution'
_w_e2solve_lineq = 'eq2solve@lineq: a=0, equation is a linear equation'
_w_e2solve_cmplx = 'eq2solve@cmplx: equation has complex solutions'

def eq2solve(a, b, c):

    if a == 0.:   # calculation or input?
        if b == 0.:
            if c == 0. :
                warnings.warn(_w_e2solve_infsol)
                return []
            warnings.warn(_w_e2solve_nosol)
            return []
        else:
            warnings.warn(_w_e2solve_lineq)
            return [-c/b]
    else:
        delta = b*b - 4.*a*c;

        if delta < 0.:
            warnings.warn(_w_e2solve_cmplx);
            x1 = (-b - 1.j * np.sqrt(abs(delta)))/(2.*a);
            x2 = (-b + 1.j * np.sqrt(abs(delta)))/(2.*a);
            return [x1, x2]
        
        delta = np.sqrt(delta);
        
        if abs(delta) <= 10**(-10):  # better than delta == 0. ?
            # delta = 0, un par de soluciones iguales;
            x1 = -b/(2.*a)
            x2 = x1
            return [x1, x2]
        elif b < 0.:        
            x1 = (-b + delta) / (2.*a)
            x2 = (2.*c) / (-b + delta)
            return [x1, x2]
        elif b > 0.:
            x1 = (-b-delta) / (2.*a)
            x2 = (2.*c) / (-b - delta)
            return [x1, x2] 
        else:
            # b = 0, un par de soluciones con signos opuestos
            x1 = sqrt(-c/a)
            x2 = -x1
            return [x1, x2]
            
            
# sqrtHeron: Computes the sqrt root of A using ancient algorithm 

# Case 1: Iterating n times, check convergence visually

def sqrtHeronTrivial(A):
	x = 1.;
	print('i = %d, root = %18.16f' % (0, x))
    
	for i in range(1,7):
		x = 0.5*(x + A/x);
        # check convergence visually
		print('i = %d, root = %18.16f' % (i, x))

	return x

# Case 2: Checking convergence automatically
    
def sqrtHeron(A, prec=16):

	delta = 10.**(-prec)
	old = 0.
	new = 1.
	count = 0

	while(abs(new-old) > delta * abs(new)):
		old = new
		new = 0.5*(new + A/new)
		count = count + 1
		
	return new, count


# Compute exp(x) using Taylor's series

def exp(x, eps=1.e-16):
    """
    This function computes the exponential function using the Taylor 
    series. It works for a scalar input x only.
	"""
	
    KMAX = 10000
    k = 1
    term = 1.
    oldsum = -1.
    newsum = 0.
	
    while ((abs(newsum-oldsum) > 0.5*abs(newsum)*eps) and (k <= KMAX)):
        oldsum = newsum
        newsum = newsum + term
        # print('tn: ', term)   # uncomment only for debug
        term = term * (x / k)
        k = k + 1
        
    if (KMAX == k-1):
        warnings.warn('Max number of iterations reached. Result could be inaccurate.')
        
    return newsum  



def horner_eval(arr, x):
    
    # Given the polynomial is a_0 + ... + a_n x^n, then
    # arr -> [a_0,...,a_n]
    
    arr = np.asarray(arr)
    x = np.asarray(x)

    # Initialize the result with the last coefficient (constant term)
    y = arr[-1]


    # Loop through the coefficients in reverse order (except the last one already used)
    for i in range(arr.size - 2, -1,-1):
        # Update the result by multiplying the current result by x and adding the next coefficient
        y = arr[i] * np.ones(x.size) + y * x #This relation also converts y into an array


    # If the result is a single value, return it directly
    if y.size == 1:
        return y[0]
    
    # If x was an array, return the computed values for each x
    return y

def binomial(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    prod = n #Since prod is already initialized at n, then we start the loop at i = 2
    for i in range(2,min(n,n-k)+1):
        prod = prod * ((n-i+1)/i)
    return prod




    
# Spherical Bessel of the first kind
    
def bessel_up(l, x):
    '''
    Upward recursion: returns array of j_i from i=0 to i=l, 
    including l
    
        Input: 
            x: scalar float
            l: integer 
        Output
            res: array of length l+1
    '''
    
    res = np.zeros(l+1)
    if abs(x)<1e-30:
        res[0] = 1.
        return res
    
    # compute j0 with known value
    j0 = np.sin(x)/x
    res[0] = j0
    if l==0: return res
    
    # compute j1 with known value
    j1 = j0/x - np.cos(x)/x
    res[1] = j1
    
    for i in range(1, l):
        j2 = (2*i + 1)/x*j1 - j0
        res[i+1] = j2
        j0, j1 = j1, j2
        
    return res
    
    
def bessel_down(l, x):
    '''
    Downward recursion: returns array of j_i from i=0 to i=l, 
    including l
    
        Input: 
            x: scalar float
            l: integer 
        Output
            res: array of length l+1    
    '''
    
    if np.abs(x) < 1e-30:
        res = np.zeros(l+1)
        res[0] = 1
        return res
    
    # heuristics to find a good l to start
    lstart = l + int(np.sqrt(10*l))
    
    # initialize j1 and j2 with any values
    j2 = 0.
    j1 = 1.
    res = []
    
    for i in range(lstart,0,-1):
        j0 = (2*i + 1)/x*j1 - j2
        if i-1<=l : 
            res.append(j0)
        j2 = j1
        j1 = j0
    
    # reverse and normalize result
    res.reverse()
    true_j0 = np.sin(x)/x
    res = np.array(res) * true_j0/res[0]
    
    return res
    
    
    
def bessel_j(l, x):
    '''
    bessel_j(l, x):  
        Combines upward and downward recursion to compute
        besselj at point for l-values: 0, 1, ..., l; i.e.
        l is the max l-value        
    '''
    
    # Case lmax <= x
    if l <= x : 
        return bessel_up(l,x)
    
    lcritical = int(x)
    
    # Case x < 1 => x < lmax (for all x) 
    if lcritical <= 0 : 
        return bessel_down(l, x)
    
    # Case x > 1 and x < lmax
    ju = bessel_up(lcritical-1, x)
    jd = bessel_down(l, x)
    
    return np.hstack( (ju, jd[lcritical:]) )    
    


# HOMEWORK

# Lentz modified algorithm for continued fractions 


def cont_frac(a, b, s = 0, x = 0, nmax = 30, tol = 1.E-8, delta = 1.E-10):
    # a, b are functions for the coefficients of the continued fraction a(n,s,x), b(n,s,x).
    # s is an additional parameter that can be passed to the coefficient functions.
    # x is the value at which the continued fraction is evaluated.
    # nmax is the maximum number of iterations.
    # tol is the tolerance for convergence.
    # delta is a small number to avoid division by zero.
    
    # Initialize f_0 according to the formula f_0 = b_0. If b_0(s,x) is zero, adjust it to delta to avoid division by zero.
    if b(0,s,x) == 0:
        fnew = delta
    else:
        fnew = b(0,s,x)  # f_0 = b_0
    
    cnew = fnew  # Initial c_0 = f_0, as per the initial condition of the algorithm.
    dnew = 0  # Initial d_0 = 0, indicating the starting value for the recurrence.
    
    for n in range(1, nmax + 1):
        # Iteratively compute f_n using the recurrence relations for c_n and d_n.
        
        # Store old values of c, d, and f to be used in the recurrence relations.
        cold = cnew
        dold = dnew
        fold = fnew
        
        # Compute a_n and b_n for the current iteration n.
        anew = a(n,s,x)
        bnew = b(n,s,x)
        
        # Compute d_n using the formula: d_n = 1 / (b_n + a_n * d_{n-1}).
        # If the denominator is too small (to avoid division by zero), adjust it to delta.
        denominatornew = bnew + anew * dold
        if abs(denominatornew) < delta:
            denominatornew = delta
        dnew = 1 / denominatornew
        
        # Compute c_n using the formula: c_n = b_n + a_n / c_{n-1}.
        # If c_n is too small (to avoid division by zero), adjust it to delta.
        cnew = bnew + anew / cold
        if abs(cnew) < delta:
            cnew = delta
        
        # Update f_n using the relation: f_n = f_{n-1} * c_n * d_n.
        fnew = fold * cnew * dnew
        
        # Check for convergence. If the relative change in f is less than the tolerance, return the current estimate.
        if abs(fnew - fold) < (0.5)*tol * abs(fnew):
            #print(fnew,fold)
            return (n,fnew)
    
    # If the loop completes without meeting the tolerance, notify the user and return the last estimate.
    #print("Tolerance not achieved")
    return (n,fnew)

# Square of two

def a_square_two(n,s,x):
    return 1

def b_square_two(n,s,x):
    return 2

def cont_frac_square_two(Nmax = 30, Tol = 1.E-8, Delta = 1.E-6):
    return cont_frac(a_square_two,b_square_two, nmax = Nmax, tol = Tol, delta = Delta) - 1






#Pi

def a_pi(n,s,x):
    if n == 1:
        return 1
    return (2 * (n-1)  -  1)** 2

def b_pi(n,s,x):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return 2

def cont_frac_pi(Nmax = 30, Tol = 1.E-8, Delta = 1.E-6):
    return cont_frac(a_pi,b_pi, nmax = Nmax, tol = Tol, delta = Delta) * 4









def b_arctan(n,s,x):
    if n == 0:
        return 0
    
    return (2 * n - 1)


def a_arctan(n,s,x):
    if n == 1:
        return 1 * x
    return ((n - 1) * x) ** 2

def cont_frac_arc_tan(X,Nmax = 30, Tol = 1.E-8, Delta = 1.E-6):
    return cont_frac(a_arctan,b_arctan,x = X, nmax = Nmax, tol = Tol, delta = Delta) 







def b_gamma(n,a,x):
    if n == 0:
        return 0
    
    if n % 2 == 0:
        return 1
    
    return x

def a_gamma(n,a,x):
    if n == 1:
        return 1
    
    if n % 2 == 0:
        return ((n // 2) - a ) 
    return ((n-2) + 1)//2

def cont_frac_inc_gamma(S,X,Nmax = 30, Tol = 1.E-8, Delta = 1.E-10):
    return ( cont_frac(a_gamma,b_gamma,s = S, x = X, nmax = Nmax, tol = Tol, delta = Delta) )  * np.exp(-X) * (X ** S)









def hermite_coeff(N):
    # Initialize a matrix to store the coefficients of Hermite polynomials up to degree N.
    # An extra column is added to the left (compared to the polynomial degree) to handle the case when k=-1,
    # which simplifies the indexing in the recursion formula.
    coeff_matrix = np.zeros((N + 1, N + 1 + 1), dtype=int)
    
    # Initial conditions for Hermite polynomials:
    # H_0(x) has a coefficient of 1 for the 0th degree (constant term),
    # and H_1(x) = 2x, so it has a coefficient of 0 for the 0th degree and 2 for the 1st degree.
    coeff_matrix[0, 0 + 1] = 1  # Setting H_0(x) = 1
    coeff_matrix[1, 0 + 1] = 0  # H_1(x) has no constant term. This is unncesarry since coeff matrix is initialized as zeros
    coeff_matrix[1, 1 + 1] = 2  # Setting the coefficient for x in H_1(x) to 2
    
    # The loop calculates coefficients for Hermite polynomials of degree 2 to N.
    for n in range(2, N + 1):
        # The recurrence relation used here is for the coefficients a_{n,k} of H_n(x):
        # a_{n,k} = 2 * a_{n-1,k-1} - 2(n-1) * a_{n-2,k}
        # This translates to shifting the coefficients of H_{n-1}(x) by one position (multiplying by x),
        # and then adjusting by the second term of the recurrence relation.
        
        # The slicing `0+1:` adjusts for the extra column at the start, intended to handle k=-1 in the formula.
        # `coeff_matrix[n-1, :-1]` selects all coefficients from the previous polynomial H_{n-1}(x),
        # shifted by one to account for multiplication by x, essentially calculating 2xH_{n-1}(x).
        coeff_matrix[n, 0+1:] = 2 * coeff_matrix[n-1, :-1]  # Calculating 2x * H_{n-1}(x)
        
        # `- 2 * (n-1) * coeff_matrix[n-2, 0+1:]` adjusts the coefficients by subtracting
        # 2(n-1) times each coefficient of H_{n-2}(x), implementing the second term of the recurrence relation.
        # This step finalizes the calculation of H_n(x) coefficients.
        coeff_matrix[n, 0+1:] -= 2 * (n-1) * coeff_matrix[n-2, 0+1:]  # Subtracting 2n * H_{n-1}(x)
    
    # Return the coefficient matrix, excluding the first column which was added for handling k=-1.
    # This matrix contains the coefficients for each Hermite polynomial H_n(x) from n=0 up to N.
    return coeff_matrix[:, 1:]





def hermite_eval(x, N):
    # Convert the input x into a numpy array to handle both single values and arrays of values.
    # This allows the function to work with multiple points in a single call.
    x = np.asarray(x)
    
    # Initialize an array H to store the evaluated values of Hermite polynomials from H_0(x) to H_N(x).
    # This will hold the final results of the polynomial evaluations.
    H = np.zeros(N+1)
    
    # Generate the coefficient matrix for Hermite polynomials up to degree N.
    # This matrix is obtained from the 'hermite_coeff' function which calculates the coefficients
    # for each Hermite polynomial up to degree N.
    coeff_matrix = hermite_coeff(N)
    
    # Loop over each degree from 0 to N to evaluate each Hermite polynomial at the given points x.
    for i in range(0, N + 1):
        # Evaluate the i-th Hermite polynomial at x using Horner's method.
        # 'horner_eval' is assumed to be a function that applies Horner's method for polynomial evaluation,
        # taking a set of coefficients (for a single polynomial) and the points x where the polynomial is to be evaluated.
        # The result is then stored in the corresponding position of the H array.
        H[i] = horner_eval(coeff_matrix[i,:], x)
    
    # Return the array of evaluated Hermite polynomials at points x.
    # H[0] corresponds to H_0(x), H[1] to H_1(x), ..., up to H[N] corresponding to H_N(x).
    return H

def hermite_poly(x,N):
    return hermite_eval(x,N)[N]


def hermite_eval_2(x,N):
    H = np.zeros(N+1)
    
    H[0] = 1
    H[1] = 2 * x
    
    for n in range(2, N+1):
        H[n] = 2 * x * H[n-1] - 2 * (n-1) * H[n-2]
    
    return H







# Function to compute the term at zero for the Laguerre polynomial
# This corresponds to the coefficient of the highest degree term, calculated as binomial(n + alpha, n)

def laguerre_term_zero(n, alpha):
    # Binomial coefficient: (n+alpha) choose (n)
    return binomial(n + alpha, n)

# Function to compute the value of the generalized Laguerre function for given n, alpha, and x
# The generalized Laguerre polynomial is defined as L_n^{(alpha)}(x)
def laguerre_fun(n, alpha, x):
    # Initial term (term at zero) calculation
    term_zero = laguerre_term_zero(n, alpha)
    # Initialize the function with the term at zero
    fun = term_zero
    
    # Initialize the 'old_term' for the recursive calculation
    old_term = term_zero
    # Loop to calculate each term in the sum for the polynomial
    for i in range(1, n + 1):
        # Recursive formula to calculate the new term based on the old term
        # New term calculation involves the polynomial's degree (n), the current index (i),
        # the function's argument (x), and the parameter (alpha)
        new_term = (n - (i - 1)) * x * old_term / ((alpha + (i - 1) + 1) * (i))
        # Update the polynomial's value by adding the new term, considering its sign
        fun += new_term * ((-1) ** i)
        # Update 'old_term' to the 'new_term' for the next iteration
        old_term = new_term
    
    # Return the calculated value of the polynomial
    return fun




# def a(n,s,x):
    #if n == 1:
        #return x
    #return x ** 2
#def b(n,s,x):
    #if n == 0:
        #return 0
    #return (2 * n - 1)

#nm.cont_frac(a, b, x = np.pi)



def cont_frac_2(a, b, s = 0, x = 0, nmax = 30, tol = 1.E-8, delta = 1.E-10, iter_out = False):
    # a, b are functions for the coefficients of the continued fraction a(n,s,x), b(n,s,x).
    # s is an additional parameter that can be passed to the coefficient functions.
    # x is the value at which the continued fraction is evaluated.
    # nmax is the maximum number of iterations.
    # tol is the tolerance for convergence.
    # delta is a small number to avoid division by zero.
    
    # Initialize f_0 according to the formula f_0 = b_0. If b_0(s,x) is zero, adjust it to delta to avoid division by zero.
    if b(0,s,x) == 0:
        fnew = delta
    else:
        fnew = b(0,s,x)  # f_0 = b_0
    
    cnew = fnew  # Initial c_0 = f_0, as per the initial condition of the algorithm.
    dnew = 0  # Initial d_0 = 0, indicating the starting value for the recurrence.
    
    for n in range(1, nmax + 1):
        # Iteratively compute f_n using the recurrence relations for c_n and d_n.
        
        # Store old values of c, d, and f to be used in the recurrence relations.
        cold = cnew
        dold = dnew
        fold = fnew
        
        # Compute a_n and b_n for the current iteration n.
        anew = a(n,s,x)
        bnew = b(n,s,x)
        
        # Compute d_n using the formula: d_n = 1 / (b_n + a_n * d_{n-1}).
        # If the denominator is too small (to avoid division by zero), adjust it to delta.
        denominatornew = bnew + anew * dold
        if abs(denominatornew) < delta:
            denominatornew = delta
        dnew = 1 / denominatornew
        
        # Compute c_n using the formula: c_n = b_n + a_n / c_{n-1}.
        # If c_n is too small (to avoid division by zero), adjust it to delta.
        cnew = bnew + anew / cold
        if abs(cnew) < delta:
            cnew = delta
        
        # Update f_n using the relation: f_n = f_{n-1} * c_n * d_n.
        fnew = fold * cnew * dnew
        
        print(fnew/fold)
        
        # Check for convergence. If the relative change in f is less than the tolerance, return the current estimate.
        if abs(fnew - fold) < (0.5)*tol * abs(fnew):
            #print(fnew,fold)
            if iter_out:
                return (n,fnew)
            return fnew
    
    # If the loop completes without meeting the tolerance, notify the user and return the last estimate.
    #print("Tolerance not achieved")
    return (n,fnew)


#PC1

def fn(n):
    if n == 0:
        return 1 - np.exp(-1)
    if n == 1:
        return 1 - 2 * np.exp(-1)
    
    else:
        return - np.exp(-1) + n * fn(n-1)
    
def fn_down(n):
    N = 2 * n
    
    fold = 0
    
    for i in range(N, n -1 , -1):
        fnew = (fold + np.exp(-1))/(i+1)
        fold = fnew
    return fnew



def legendre_coeff(N):

    coeff_matrix = np.zeros((N + 1, N + 1 + 1), dtype=float)
    

    coeff_matrix[0, 0 + 1] = 1  
    coeff_matrix[1, 0 + 1] = 0  
    coeff_matrix[1, 1 + 1] = 1
    
 
    for n in range(2, N + 1):
       
        coeff_matrix[n, 0+1:] = ((2*n-1)/n) * coeff_matrix[n-1, :-1] 
        
     
        coeff_matrix[n, 0+1:] -= ((n-1)/n) * coeff_matrix[n-2, 0+1:] 
    
    
    return coeff_matrix[:, 1:]




def legendre_eval(x, N):

    x = np.asarray(x, dtype = float)
    

    H = np.zeros(N+1, dtype = float)
    
 
    coeff_matrix = legendre_coeff(N)
    

    for i in range(0, N + 1):
        H[i] = horner_eval(coeff_matrix[i,:], x)
    
    return H

def legendre_poly(x,N):
    return legendre_eval(x, N)[N]

def der_legendre_poly(x, N):
    return (N * legendre_poly(x,N-1) - N * x * legendre_poly(x,N)) / (1 - x**2)

def legendre_eval_2(x,N):
    H = np.zeros(N+1, dtype = float)
    
    H[0] = 1
    H[1] = x
    
    for n in range(2, N+1):
        H[n] = ((2 * n - 1) * x * H[n-1] - (n-1) * H[n-2])/(n)
    
    return H
