import numpy as np


def der1data(x, y, h, method):
    '''DER1DATA computes the first derivative of 
    sampled function (X,Y).
    DER1DATA(X,Y,H,METHOD) computes de first derivative 
    of the sampled function given as X and Y arrays. It 
    is assumed that, the abscisas X are equally spaced 
    with step H. The possible methods are:
    'f': forward differences with one step
    'b': backward differences with one step
    'c': central differences with one step'''
    
    x = np.asarray(x)
    y = np.asarray(y)
    
    
    if method == 'f':
        df = (y[1:] - y[0:-1]) / h
        x = x[0:-1]
    elif method == 'b':
        df = (y[1:] - y[0:-1]) / h
        x = x[1:]
    elif method =='c':
        df = (y[2:] - y[0:-2]) / (2*h)
        x = x[1:-1]
    else:
        print('invalid method');
    
    return np.array([x, df])
    
    
    
def der1(f, x, h, method):
    '''DER1 computes the first derivative of function F.
    DER1DATA(F,X,H,METHOD) computes de first derivative 
    of the function F at point X. The possible methods are:
    'f': forward differences with one step
    'b': backward differences with one step
    'c': central differences with one step.
    X could be an array or instead H could be an array,
    but they should be created using numpy.'''
	
    if method == 'f':
        df = (f(x + h) - f(x)) / h
    elif method == 'b':
        df = (f(x) - f(x - h)) / h
    elif method =='c':
        df = (f(x + h) - f(x - h)) / (2.*h)
    else:
        print('invalid method');
        
    return df



# First derivative using O[h**2] central formula and a simple
# method to find the optimal step size
  
def der_min_1(f, x, h = 0.1):
    '''DER_MIN_1 computes the first derivative of function F.
    DER_MIN_1(F,X,H,METHOD) computes de first derivative of the 
    function F at point X using the method of the minimum error. 
    An estimate of the initial step can be given'''
    
    c = 1.4

    dfold = (f(x + h) - f(x - h)) / (2.*h)
    h = h/c
    dfnew = (f(x + h) - f(x - h)) / (2.*h)
    
    err_old = 1e30
    err_new = abs(dfnew - dfold)

    while( err_old > err_new ):
        
        dfold = dfnew
        h = h/c			
        dfnew = (f(x + h) - f(x - h)) / (2.*h)
        
        err_old = err_new
        err_new = abs(dfnew - dfold)

    return dfold, err_old



def CD(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def richardson_der(f, x, h=0.1, r=1.4, n=8, debug = False):

    # Initialize the differences matrix
    D = np.zeros((n, n), dtype = np.float64)
    E = np.zeros((n, n), dtype = np.float64)
    
    
    D[0,0] = CD(f,x,h)
    E[0,0] = 10 ** 30
    error_max = E[0,0]
    drow = D[0,0]
    drowold = drow
    error_max_old = error_max
    for i in range(1,n):
        h = h / r
        D[i,0] = CD(f,x,h)
        E[i,0] = abs(D[i,0]-D[i-1,0])
        
        for j in range(1,i+1):
            D[i,j] = D[i-1,j-1] - ((r**2)**j)*D[i,j-1]
            D[i,j] = D[i,j] / (1-((r**2)**j))
            E[i,j] = max(abs(D[i,j]-D[i-1,j-1]),abs(D[i,j]-D[i,j-1]))
            error = E[i,j]
            #print(error,error_max)
            if error < error_max:
                error_max_old = error_max
                error_max = error
                drowold = drow
                drow = D[i,j]
        
        #print("error",error)
            #print("dold",drowold)
            #print("d",drow)
        
        if (abs(drow - drowold) >= 2.0 * error_max or abs(D[i,i] - D[i-1,i-1]) >= 2.0 * error_max): 
            #print(D[i-2:,i-2:])
            #print(E[i-2:,i-2:])
            #print("drowold ", drowold, "drow ", drow)
            #print("Error max old ",error_max_old)
            #print("Exit condition. Error: ", error_max, "drow-drowold: ", abs(drow - drowold) )
            #print(drow,drowold)
            #print(D[i,i],D[i-1,i-1])
            #print("EXIT EARLY")
            if debug:
                print(D)
            return drowold, error_max
        
        
            
    #print(E[2:,2:])
    if debug:
        print(D)
    #print("accu", D[n-1,n-1])
    return drow,error_max


#OWN FUNCTIONS


def CD2(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x-h)) / (h**2)

def richardson_der2(f, x, h=0.1, r=1.4, n=8, debug = False):

    # Initialize the differences matrix
    D = np.zeros((n, n), dtype = np.float64)
    E = np.zeros((n, n), dtype = np.float64)
    
    
    D[0,0] = CD2(f,x,h)
    E[0,0] = 10 ** 30
    error_max = E[0,0]
    drow = D[0,0]
    drowold = drow
    error_max_old = error_max
    for i in range(1,n):
        h = h / r
        D[i,0] = CD2(f,x,h)
        E[i,0] = abs(D[i,0]-D[i-1,0])
        
        for j in range(1,i+1):
            D[i,j] = D[i-1,j-1] - ((r**2)**j)*D[i,j-1]
            D[i,j] = D[i,j] / (1-((r**2)**j))
            E[i,j] = max(abs(D[i,j]-D[i-1,j-1]),abs(D[i,j]-D[i,j-1]))
            error = E[i,j]
            #print(error,error_max)
            if error < error_max:
                error_max_old = error_max
                error_max = error
                drowold = drow
                drow = D[i,j]
        
        #print("error",error)
            #print("dold",drowold)
            #print("d",drow)
        
        if (abs(drow - drowold) >= 2.0 * error_max or abs(D[i,i] - D[i-1,i-1]) >= 2.0 * error_max): 
            #print(D[i-2:,i-2:])
            #print(E[i-2:,i-2:])
            #print("drowold ", drowold, "drow ", drow)
            #print("Error max old ",error_max_old)
            #print("Exit condition. Error: ", error_max, "drow-drowold: ", abs(drow - drowold) )
            #print(drow,drowold)
            #print(D[i,i],D[i-1,i-1])
            #print("EXIT EARLY")
            if debug:
                print(D)
            return drowold, error_max
        
        
            
    #print(E[2:,2:])
    if debug:
        print(D)
    #print("accu", D[n-1,n-1])
    return drow,error_max




def der3p(f, x, h, method='f'):
    """
    Calculates the derivative of a function f at a point x using the 3-point formula (completely non-vectorized).
    
    Parameters:
    f : function
        The function to be derived.
    x : float
        The point at which to calculate the derivative.
    h : float
        Step size for the difference calculation.
    method : str, optional
        Differentiation method: 'f' for forward, 'b' for backward.
    
    Returns:
    dy : float
        Derivative of f at point x.
    """
    dy = 0

    if method == 'f':
        # Three-point forward formula
        dy = (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
    elif method == 'b':
        # Three-point backward formula
        dy = (f(x - 2 * h) - 4 * f(x - h) + 3 * f(x)) / (2 * h)
    
    return dy


